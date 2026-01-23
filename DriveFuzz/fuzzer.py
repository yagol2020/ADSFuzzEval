#!/usr/bin/env python3
import copyreg
import os
import sys
import time
import random
import argparse
import json
from collections import deque
import math
import datetime
import signal
from copy import deepcopy
import traceback
import shutil
from opcode import opname

import dill
import timeout_decorator.timeout_decorator

import config
import constants as c
import states

try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

import fuzz_utils


def handler(signum, frame):
    raise Exception("HANG")


def init(conf, args):
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)

    conf.out_dir = args.out_dir
    try:
        os.mkdir(conf.out_dir)
    except Exception as e:
        estr = f"Output directory {conf.out_dir} already exists. Remove with " \
               "caution; it might contain data from previous runs."
        print(estr)
        if len(os.listdir(conf.out_dir)) > 0:
            print(f"Directory {conf.out_dir} is not empty. Aborting.")
            sys.exit(-1)
        else:
            print(f"Directory {conf.out_dir} exists but is empty. Continuing...")

    conf.seed_dir = args.seed_dir
    if not os.path.exists(conf.seed_dir):
        os.mkdir(conf.seed_dir)
    else:
        print(f"Using seed dir {conf.seed_dir}")

    if args.verbose:
        conf.debug = True
    else:
        conf.debug = False

    conf.set_paths()

    with open(conf.meta_file, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
        f.write("start: " + str(int(conf.cur_time)) + "\n")

    try:
        os.mkdir(conf.queue_dir)
        os.mkdir(conf.error_dir)
        os.mkdir(conf.cov_dir)
        os.mkdir(conf.rosbag_dir)
        os.mkdir(conf.cam_dir)
        os.mkdir(conf.score_dir)
    except Exception as e:
        print(e)
        sys.exit(-1)

    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port

    conf.max_cycles = args.max_cycles
    conf.max_mutations = args.max_mutations

    conf.timeout = args.timeout

    conf.function = args.function

    if args.target.lower() == "basic":
        conf.agent_type = c.BASIC
    elif args.target.lower() == "behavior":
        conf.agent_type = c.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = c.AUTOWARE
    elif args.target.lower() == "interfuser":
        conf.agent_type = c.INTERFUSER
    elif args.target.lower() == "lmdrive":
        conf.agent_type = c.LMDRIVE
    elif args.target.lower() == "autoware_0903":
        conf.agent_type = c.AUTOWARE_0903
        import autoware_2409_utils

        autoware_2409_utils.zero_counters()
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)
    print("[info] target set to:", conf.agent_type, args.target.lower())

    conf.town = args.town

    if args.no_speed_check:
        conf.check_dict["speed"] = False
    if args.no_crash_check:
        conf.check_dict["crash"] = False
    if args.no_lane_check:
        conf.check_dict["lane"] = False
    if args.no_stuck_check:
        conf.check_dict["stuck"] = False
    if args.no_red_check:
        conf.check_dict["red"] = False
    if args.no_other_check:
        conf.check_dict["other"] = False

    if args.strategy == "all":
        conf.strategy = c.ALL
    elif args.strategy == "congestion":
        conf.strategy = c.CONGESTION
    elif args.strategy == "entropy":
        conf.strategy = c.ENTROPY
    elif args.strategy == "instability":
        conf.strategy = c.INSTABILITY
    elif args.strategy == "trajectory":
        conf.strategy = c.TRAJECTORY
    else:
        print("[-] Please specify a strategy")
        exit(-1)


def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["puddle"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)


def mutate_weather_fixed(test_scenario):
    test_scenario.weather["cloud"] = 0
    test_scenario.weather["rain"] = 0
    test_scenario.weather["puddle"] = 0
    test_scenario.weather["wind"] = 0
    test_scenario.weather["fog"] = 0
    test_scenario.weather["wetness"] = 0
    test_scenario.weather["angle"] = 0
    test_scenario.weather["altitude"] = 60


def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--out-dir", default="output", type=str,
                           help="Directory to save fuzzing logs")
    argparser.add_argument("-s", "--seed-dir", default="../seed_scenarios1/Town03", type=str,
                           help="Seed directory")
    argparser.add_argument("-c", "--max-cycles", default=10, type=int,
                           help="Maximum number of loops")
    argparser.add_argument("-m", "--max-mutations", default=10, type=int,
                           help="Size of the mutated population per cycle")
    argparser.add_argument("-d", "--determ-seed", type=float,
                           help="Set seed num for deterministic mutation (e.g., for replaying)")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           default=False, help="enable debug mode")
    argparser.add_argument("-u", "--sim-host", default="localhost", type=str,
                           help="Hostname of Carla simulation server")
    argparser.add_argument("-p", "--sim-port", default=2000, type=int,
                           help="RPC port of Carla simulation server")
    argparser.add_argument("-t", "--target", default="interfuser", type=str,
                           help="Target autonomous driving system (basic/behavior/autoware)")
    argparser.add_argument("-f", "--function", default="general", type=str,
                           choices=["general", "collision", "traction", "eval-os", "eval-us",
                                    "figure", "sens1", "sens2", "lat", "rear"],
                           help="Functionality to test (general / collision / traction)")
    argparser.add_argument("--strategy", default="all", type=str,
                           help="Input mutation strategy (all / congestion / entropy / instability / trajectory)")
    argparser.add_argument("--town", default=3, type=int,
                           help="Test on a specific town (e.g., '--town 3' forces Town03)")
    argparser.add_argument("--timeout", default="60", type=int,
                           help="Seconds to timeout if vehicle is not moving")
    argparser.add_argument("--no-speed-check", action="store_true")
    argparser.add_argument("--no-lane-check", action="store_true")
    argparser.add_argument("--no-crash-check", action="store_true")
    argparser.add_argument("--no-stuck-check", action="store_true")
    argparser.add_argument("--no-red-check", action="store_true")
    argparser.add_argument("--no-other-check", action="store_true")

    return argparser


def main():
    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)

    cov_dict = {}

    queue = deque(conf.enqueue_seed_scenarios())

    scene_id = 0
    campaign_cnt = 0

    signal.signal(signal.SIGALRM, handler)

    json.dump(time.time(), open(conf.out_dir + "/start_time.json", "w"))

    while True:
        cycle_cnt = 0
        try:
            scenario = queue.popleft()
            conf.cur_scenario = scenario
            scene_id += 1
            campaign_cnt += 1

            print("\n\033[1m\033[92m" + "=" * 10 + " NEW CAMPAIGN " + "=" * 10 + "\033[0m\n")

        except IndexError:
            print("[-] Seed queue is empty. Continue with random seed.")

            if conf.town is not None:
                town_map = "Town0{}".format(conf.town)
            else:
                town_map = "Town0{}".format(random.randint(1, 2))
            print(f"Loading {town_map}.sps.pick")
            spawn_points = dill.load(open(f"town_sps/{town_map}.sps.pick", "rb"))

            sp = random.choice(spawn_points)
            sp_x = sp.location.x
            sp_y = sp.location.y
            sp_z = sp.location.z
            pitch = sp.rotation.pitch
            yaw = sp.rotation.yaw
            roll = sp.rotation.roll

            wp = random.choice(spawn_points)
            wp_x = wp.location.x
            wp_y = wp.location.y
            wp_z = wp.location.z
            wp_yaw = wp.rotation.yaw

            # restrict destinition to be within 200 meters
            while math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > 100:
                wp = random.choice(spawn_points)
                wp_x = wp.location.x
                wp_y = wp.location.y
                wp_z = wp.location.z
                wp_yaw = wp.rotation.yaw

            seed_dict = {
                "map": town_map,
                "sp_x": sp_x,
                "sp_y": sp_y,
                "sp_z": sp_z,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "wp_x": wp_x,
                "wp_y": wp_y,
                "wp_z": wp_z,
                "wp_yaw": wp_yaw
            }
            scene_name = "scene-created{}.json".format(scene_id)
            with open(os.path.join(conf.seed_dir, scene_name), "w") as fp:
                json.dump(seed_dict, fp)
            queue.append(scene_name)

            continue

        if conf.debug:
            print("[*] USING SEED FILE:", scenario)

        # STEP 2: TEST CASE INITIALIZATION
        # Create and initialize a TestScenario instance based on the metadata
        # read from the popped seed file.

        test_scenario = fuzz_utils.TestScenario(conf)
        successor_scenario = None
        first_seed = True

        # STEP 3: SCENE MUTATION
        # While test scenario has quota left, mutate scene by randomly
        # generating walker/vehicle/puddle.

        while cycle_cnt < conf.max_cycles:
            cycle_cnt += 1

            if successor_scenario is not None:
                test_scenario = successor_scenario

            mutated_scenario_list = list()  # mutated TestScenario objects
            score_list = list()  # driving scores of each mutated scenario

            round_cnt = 0
            town = test_scenario.town
            (min_x, max_x, min_y, max_y) = fuzz_utils.get_valid_xy_range(town)

            while round_cnt < conf.max_mutations:  # mutation rounds

                round_cnt += 1

                print("\n\033[1m\033[92mCampaign #{} Cycle #{}/{} Mutation #{}/{}".format(
                    campaign_cnt, cycle_cnt, conf.max_cycles, round_cnt,
                    conf.max_mutations), "\033[0m", datetime.datetime.now())

                test_scenario_m = deepcopy(test_scenario)
                mutated_scenario_list.append(test_scenario_m)

                # STEP 3-1: ACTOR PROFILE GENERATION

                if conf.function == "general":

                    if conf.strategy == c.ALL:
                        actor_type = random.randint(0, len(c.ACTOR_LIST) - 1)
                        nav_type = random.randint(0, len(c.NAVTYPE_LIST) - 1)

                    elif conf.strategy == c.CONGESTION:
                        actor_type = random.randint(0, len(c.ACTOR_LIST) - 1)
                        nav_type = c.AUTOPILOT

                    elif conf.strategy == c.ENTROPY:
                        actor_type = random.randint(0, len(c.ACTOR_LIST) - 1)
                        nav_type = c.LINEAR

                    elif conf.strategy == c.INSTABILITY:
                        actor_type = c.NULL
                        nav_type = c.NULL

                    elif conf.strategy == c.TRAJECTORY:
                        actor_type = c.VEHICLE
                        nav_type = c.MANEUVER

                elif conf.function == "collision":
                    actor_type = c.ACTOR_LIST[c.VEHICLE]
                    nav_type = c.NAVTYPE_LIST[c.LINEAR]

                elif conf.function == "traction":
                    actor_type = c.NULL
                    nav_type = c.NULL

                ret = True
                while ret:
                    if actor_type == c.NULL:
                        break

                    elif actor_type == c.VEHICLE:
                        # spawn vehicle at a random location
                        # run test while mutating weather
                        if nav_type == c.LINEAR:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = random.uniform(0, c.VEHICLE_MAX_SPEED)

                            loc = (x, y, z)  # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw)  # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type,
                                                            nav_type, loc, rot, speed, None, None)

                        elif nav_type == c.IMMOBILE:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = 0

                            loc = (x, y, z)  # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw)  # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type,
                                                            nav_type, loc, rot, speed, None, None)

                        elif nav_type == c.AUTOPILOT:
                            sp_idx = random.randint(0, c.NUM_WAYPOINTS[town] - 1)
                            dp_idx = random.randint(0, c.NUM_WAYPOINTS[town] - 1)
                            ret = test_scenario_m.add_actor(actor_type,
                                                            nav_type, None, None, None, sp_idx,
                                                            dp_idx)

                        elif nav_type == c.MANEUVER:
                            if len(test_scenario_m.actors) == 0:
                                # add an actor
                                x = test_scenario_m.sp.location.x + random.randint(-50, 50) / 10.0

                                y = test_scenario_m.sp.location.y + random.randint(-50, 50) / 10.0
                                z = 1.5

                                roll = test_scenario_m.sp.rotation.roll
                                pitch = test_scenario_m.sp.rotation.pitch
                                yaw = test_scenario_m.sp.rotation.yaw

                                speed = 0

                                loc = (x, y, z)  # carla.Location(x, y, z)
                                rot = (roll, pitch, yaw)  # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                                ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                                                speed, None, None)

                            else:
                                # mutate maneuvers if we have an actor
                                i = random.randint(0, 4)
                                direction = random.randint(-1, 1)

                                if direction == 0:
                                    speed = random.randint(0, 10)  # m/s
                                    test_scenario_m.actors[0]["maneuvers"][i] = [direction, speed, 0]
                                else:
                                    degree = random.randint(30, 60)  # deg
                                    test_scenario_m.actors[0]["maneuvers"][i] = [direction, degree, 0]

                                # reset executed frame id
                                for i in range(5):
                                    test_scenario_m.actors[0]["maneuvers"][i][2] = 0

                                ret = 0

                            print("Maneuvers:", test_scenario_m.actors[0]["maneuvers"])

                    elif actor_type == c.WALKER:
                        if nav_type == c.LINEAR:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = random.uniform(0, c.WALKER_MAX_SPEED)

                            loc = (x, y, z)  # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw)  # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                                            speed, None, None)

                        elif nav_type == c.IMMOBILE:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = 0

                            loc = (x, y, z)  # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw)  # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                                            speed, None, None)

                        elif nav_type == c.AUTOPILOT:
                            sp_idx = random.randint(0, c.NUM_WAYPOINTS[town] - 1)
                            dp_idx = random.randint(0, c.NUM_WAYPOINTS[town] - 1)
                            speed = random.randint(0, 10)
                            try:  # error in add_actor, skip if stuck too long
                                ret = test_scenario_m.add_actor(actor_type, nav_type, None,
                                                                None, speed, sp_idx, dp_idx)
                            except timeout_decorator.timeout_decorator.TimeoutError:
                                ret = 0

                if conf.debug:
                    if actor_type != c.NULL:
                        print("[debug] successfully added {} {}".format(
                            c.NAVTYPE_NAMES[nav_type], c.ACTOR_NAMES[actor_type]))

                # STEP 3-2: PUDDLE PROFILE GENERATION
                if conf.function == "traction":
                    prob_puddle = 0  # always add a puddle
                elif conf.function == "collision":
                    prob_puddle = 100  # never add a puddle
                elif conf.strategy == c.INSTABILITY:
                    prob_puddle = 0
                else:
                    prob_puddle = random.randint(0, 100)

                if (prob_puddle < c.PROB_PUDDLE):
                    ret = True
                    while ret:
                        # slippery puddles only
                        level = random.randint(0, 200) / 100
                        x = random.randint(min_x, max_x)
                        y = random.randint(min_y, max_y)
                        z = 0

                        xy_size = c.PUDDLE_MAX_SIZE
                        z_size = 1000  # doesn't affect anything

                        loc = (x, y, z)  # carla.Location(x, y, z)
                        size = (xy_size, xy_size, z_size)  # carla.Location(xy_size, xy_size, z_size)
                        ret = test_scenario_m.add_puddle(level, loc, size)

                    if conf.debug:
                        print("successfully added a puddle")

                # print("after seed gen and mutation", time.time())
                # STEP 3-3: EXECUTE SIMULATION
                ret = None

                state = states.State()
                state.campaign_cnt = campaign_cnt
                state.cycle_cnt = cycle_cnt
                state.mutation = round_cnt
                state.first_seed = first_seed

                mutate_weather(test_scenario_m)  # mutate_weather_fixed(test)

                signal.alarm(15 * 60)  # timeout after 10 mins
                try:
                    ret = test_scenario_m.run_test(state)
                    first_seed = False
                except Exception as e:
                    if e.args[0] == "HANG":
                        print("[-] simulation hanging. abort.")
                        ret = -1
                    else:
                        print("[-] run_test error:")
                        traceback.print_exc()
                signal.alarm(0)

                # t3 = time.time()

                # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
                # TODO: map return codes with failures, e.g., spawn
                if ret is None:
                    # failure
                    pass
                elif ret == -2:
                    # error in simulation
                    continue

                elif ret == -1:
                    print("Spawn / simulation failure - don't add round cnt")
                    round_cnt -= 1

                elif ret == 1:
                    print("fuzzer - found an error")
                    # found an error - move on to next one in queue
                    # test.quota = 0
                    break

                elif ret == 128:
                    print("Exit by user request")
                    exit(0)

                else:
                    if ret == -1:
                        print("[-] Fatal error occurred during test")
                        exit(-1)

                score_list.append(test_scenario_m.driving_quality_score)
                ### mutation loop ends

            if test_scenario_m.found_error:
                # error detected. start a new cycle with a new seed
                successor_scenario = test_scenario_m
                break
            try:
                idx = score_list.index(min(score_list))
            except:
                idx = 0
            successor_scenario = mutated_scenario_list[idx]

            print(score_list)
            print(mutated_scenario_list)
            print("successor:", vars(successor_scenario))

            shutil.copyfile(
                os.path.join(conf.queue_dir, successor_scenario.log_filename),
                os.path.join(conf.cov_dir, successor_scenario.log_filename)
            )

            print("=" * 10 + " END OF ALL CYCLES " + "=" * 10)


def location_pickle(location):
    return carla.Location, (location.x, location.y, location.z)


def rotation_pickle(rotation):
    return carla.Rotation, (rotation.pitch, rotation.yaw, rotation.roll)


def transform_pickle(transform):
    return carla.Transform, (
        carla.Location(transform.location.x, transform.location.y, transform.location.z),
        carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
    )


if __name__ == "__main__":
    copyreg.pickle(carla.Location, location_pickle)
    copyreg.pickle(carla.Rotation, rotation_pickle)
    copyreg.pickle(carla.Transform, transform_pickle)
    main()
