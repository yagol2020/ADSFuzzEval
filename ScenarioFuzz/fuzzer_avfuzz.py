#!/usr/bin/env python3
import copyreg
import os
import sys, copy
import time
import random
import argparse, pickle
import json
import datetime
import signal
from copy import deepcopy
import traceback
import shutil
import warnings

import config
import constants as C
import states
from sample_config.sample_config import ValueRangeManager
import fuzz_utils

import scenario_select.scenario_selector as scenario_selector

from avfuzzer.GeneticAlgorithm import GeneticAlgorithm
from avfuzzer.cal import cal_fitness
import avfuzzer.generateRestart as generateRestart
from avfuzzer.chromosome import Chromosome
import carla

weather_list = ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning)

# 忽略所有警告
warnings.filterwarnings('ignore')


def random_select_point(town_str):
    sps_json = json.load(open(f"./towns_map/Town0{town_str}.wps.json"))
    road_id = random.choice(list(sps_json.keys()))
    point = random.choice(sps_json[road_id])
    format_point = {
        "x": point["location"]["x"],
        "y": point["location"]["y"],
        "z": point["location"]["z"],
        "roll": point["rotation"]["roll"],
        "pitch": point["rotation"]["pitch"],
        "yaw": point["rotation"]["yaw"],
    }
    return format_point


class Chromosome_sf(Chromosome):
    def __init__(self, speed_bounds, action_bounds, NPC_size, time_size):
        super().__init__(speed_bounds, action_bounds, NPC_size, time_size)
        self.campaign_cnt = 0
        self.cycle_cnt = 0
        self.mutation = 0

    def func(self, gen=None, lisFlag=False):

        global conf, args, alarm_time, test_scenario, scenario_dict

        self.cycle_cnt = gen

        mutation_start_time = datetime.datetime.now()
        conf.time_list['mutation_start_time'] = mutation_start_time.strftime('%Y-%m-%d %H:%M:%S')
        signal.alarm(alarm_time * 60)

        test_scenario_m = deepcopy(test_scenario)
        ego_car_start_point = scenario_dict['ego']['start_point']
        ego_car_end_point = scenario_dict['ego']['end_point']

        ego_s = [(ego_car_start_point['x'], ego_car_start_point['y'], max(2, ego_car_start_point['z'] + 1)),
                 (ego_car_start_point['roll'], ego_car_start_point['pitch'], ego_car_start_point['yaw'])]
        ego_w = [(ego_car_end_point['x'], ego_car_end_point['y'], max(2, ego_car_end_point['z'] + 1)),
                 (ego_car_end_point['roll'], ego_car_end_point['pitch'], ego_car_end_point['yaw'])]
        test_scenario_m.set_ego_wp_sp(ego_s, ego_w, specific_waypoints=True)

        npcs_start_point = scenario_dict['other_vehicle']

        maneuver_object = self.scenario
        numOfNpc = len(maneuver_object)
        for npc_index in range(numOfNpc):
            actor_type = C.VEHICLE
            nav_type = C.MANEUVER

            npc_start_point = npcs_start_point[npc_index]['start_point']

            npc_s = [(npc_start_point['x'], npc_start_point['y'], max(2, npc_start_point['z'] + 1)),
                     (npc_start_point['roll'], npc_start_point['pitch'], npc_start_point['yaw'])]

            nps_manuever = maneuver_object[npc_index]

            ret = test_scenario_m.add_actor(actor_type, nav_type, npc_s, None, 0, name="vehicle.bmw.grandtourer",
                                            maneuvers=nps_manuever, specific_waypoints=True)

        # 执行场景
        ret = None
        # time.sleep(20)
        state = states.State()
        state.campaign_cnt = self.campaign_cnt
        state.cycle_cnt = self.cycle_cnt
        state.mutation = self.mutation
        state.file_name = "{}_{}_{}".format(state.campaign_cnt, state.cycle_cnt, state.mutation)

        state.device = args.device

        exec_start_time = datetime.datetime.now()
        state.exec_start_time = exec_start_time.strftime('%Y-%m-%d %H:%M:%S')

        try:
            ret, state = test_scenario_m.run_test(state)
            print(ret, state)
        except Exception as e:
            if e.args[0] == "HANG":
                print("[-] simulation hanging. abort.")
            else:
                print("[-] run_test error:")
            traceback.print_exc()
            ret = -1
        signal.alarm(0)

        # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
        # TODO: map return codes with failures, e.g., spawn
        if ret is None or ret == 0:
            pass

        elif ret == -1:
            print("Spawn / simulation failure - don't add round cnt")
            return

        elif ret == 1:
            print("fuzzer - found an error")

        elif ret == 128:
            print("Exit by user request")
            exit(0)

        elif ret == 666:
            print("each the end point but systems no ending")

        else:
            print("[-] Fatal error occurred during test")
            sys.exit(-1)

        ego_car_trajectory = state.object_trajectory['ego_car']['trajectory']
        ego_speed = state.speed
        other_vehicles_trajectories = [v['trajectory'] for v in state.object_trajectory['vehicles'].values()]
        fitness = cal_fitness(ego_car_trajectory, other_vehicles_trajectories, ego_speed, numOfNpc)
        self.y = fitness

        # 存储fitness
        fitness_file_name = state.file_name + '-{}.json'.format(time.time())

        fitness_file_path = os.path.join(conf.scenario_data, fitness_file_name)
        fitness_file = open(fitness_file_path, 'w')
        fitness_file.write(json.dumps(fitness))
        fitness_file.close()

        print("fitness:", fitness)


def handler(signum, frame):
    print("[-] 代码运行超时，结束运行。")
    raise Exception("HANG")


def timeout_handler(signum, frame):
    print("[-] 代码运行超时，结束运行。")
    sys.exit(-1)


def init(conf, args):
    global NUM_SAMPLE
    ''' 
    参数赋值于conf类,创建文件夹
    
    '''
    # Fuzzing 开始时间
    NUM_SAMPLE = args.mutation_num
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)

    target_dir = args.out_dir + '/{}'.format(args.target).replace(':', '-')
    os.makedirs(target_dir, exist_ok=True)

    now = datetime.datetime.now()
    time_now = now.strftime("%Y-%m-%d-%H-%M")
    out_dir = os.path.join(target_dir, 'out-artifact-{}'.format(time_now))
    # print(target_dir,out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)
    conf.out_dir = out_dir
    print("[info] all information wiil be stored at:", conf.out_dir)

    if args.seed_dir is None:
        conf.seed_dir = os.path.join(target_dir, 'seed-artifact')
    else:
        conf.seed_dir = args.seed_dir

    conf.cache_dir = args.cache_dir
    if not os.path.exists(conf.cache_dir):
        os.mkdir(conf.cache_dir)
    print(f"Using cache dir {conf.cache_dir}")

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
        os.mkdir(conf.scenario_data)
    except Exception as e:
        print(e)
        sys.exit(-1)

    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port
    conf.sim_tm_port = conf.sim_port + 50
    conf.max_cycles = args.max_cycles
    conf.max_mutations = args.max_mutations

    conf.timeout = args.timeout

    conf.function = args.function

    if args.target.lower() == "basic":
        conf.agent_type = C.BASIC
    elif args.target.lower() == "behavior":
        conf.agent_type = C.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = C.AUTOWARE
    elif args.target.lower() == "interfuser":
        conf.agent_type = C.INTERFUSER
    elif args.target.lower() == "autoware_0903":
        conf.agent_type = C.AUTOWARE_0903
        import autoware_2409_utils

        autoware_2409_utils.zero_counters()
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)
    # 若添加新系统，需从此入门修改。
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
        conf.strategy = C.ALL
    elif args.strategy == "congestion":
        conf.strategy = C.CONGESTION
    elif args.strategy == "entropy":
        conf.strategy = C.ENTROPY
    elif args.strategy == "instability":
        conf.strategy = C.INSTABILITY
    elif args.strategy == "trajectory":
        conf.strategy = C.TRAJECTORY
    else:
        print("[-] Please specify a strategy")
        exit(-1)

    if args.coverage:
        conf.cov_mode = True


def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["puddle"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)


def set_weather(test_scenario, weather_param):
    test_scenario.weather["cloud"] = weather_param["cloud"]
    test_scenario.weather["rain"] = weather_param["rain"]
    test_scenario.weather["puddle"] = weather_param["puddle"]
    test_scenario.weather["wind"] = weather_param["wind"]
    test_scenario.weather["fog"] = weather_param["fog"]
    test_scenario.weather["wetness"] = weather_param["wetness"]
    test_scenario.weather["angle"] = weather_param["angle"]
    test_scenario.weather["altitude"] = weather_param["altitude"]


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
    argparser.add_argument("-o", "--out-dir", default="./avfuzzer_data", type=str,
                           help="Directory to save fuzzing logs")
    argparser.add_argument("--cache-dir", default="/tmp/fuzzerdata-avfuzzer", type=str,
                           help="store fuzz cache data")
    argparser.add_argument("--scenario-lib-dir", default="scenario_lib", type=str,
                           help="store scenario lib  data")
    argparser.add_argument("-s", "--seed-dir", default=None, type=str,
                           help="Seed directory")
    argparser.add_argument("-c", "--max-cycles", default=3, type=int,
                           help="Maximum number of loops")
    argparser.add_argument("-m", "--max-mutations", default=2, type=int,
                           help="Size of the mutated population per cycle")
    argparser.add_argument("--mutation-num", default=100, type=int, help="Size of the mutated population per cycle")
    argparser.add_argument("-d", "--determ-seed", type=float,
                           help="Set seed num for deterministic mutation (e.g., for replaying)")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           default=False, help="enable debug mode")
    argparser.add_argument("--direction-set", action="store_true",
                           default=False, help="enable debug mode")
    argparser.add_argument("-u", "--sim-host", default="localhost", type=str,
                           help="Hostname of Carla simulation server")
    argparser.add_argument("-p", "--sim-port", default=5000, type=int,
                           help="RPC port of Carla simulation server")
    argparser.add_argument("-t", "--target", default="interfuser", type=str,
                           help="Target autonomous driving system (basic/behavior/autoware/leaderboard:(Transfuser/NEAT/LAV/))")
    argparser.add_argument("-f", "--function", default="general", type=str,
                           choices=["general", "collision", "traction", "eval-os", "eval-us",
                                    "figure", "sens1", "sens2", "lat", "rear"],
                           help="Functionality to test (general / collision / traction)")
    argparser.add_argument("--strategy", default="all", type=str,
                           help="Input mutation strategy (all / congestion / entropy / instability / trajectory)")
    argparser.add_argument("--town", default='Town03', type=str,
                           help="Test on a specific town (e.g., '--town 3' forces Town03),all will be random choose")

    argparser.add_argument("--scenario-id", default=3, type=int,
                           help="Test on a specific scenario id (e.g., '--scenario-id 3' forces scenario 3)")

    argparser.add_argument("--timeout", default="60", type=int,
                           help="Seconds to timeout if vehicle is not moving")
    argparser.add_argument("-a", "--alarm-time", default="20", type=int,
                           help="Seconds to timeout if vehicle is not moving")

    argparser.add_argument("--eval-model", default="improve-v2", type=str,
                           help="set_eval_model_type", choices=["improve", "basic", "improve-v1", "improve-v2"])
    argparser.add_argument("--device", default="cuda:0", type=str,
                           help="device to run the model(default:cuda:0/1,cpu)")
    argparser.add_argument("--no-use-seed", action="store_true", help="no use seed in first cycle")
    argparser.add_argument("--no-speed-check", action="store_true")
    argparser.add_argument("--no-lane-check", action="store_true")
    argparser.add_argument("--no-crash-check", action="store_true")
    argparser.add_argument("--no-stuck-check", default=False, action="store_true")
    argparser.add_argument("--no-red-check", action="store_true")
    argparser.add_argument("--no-other-check", default=True, action="store_true")
    argparser.add_argument("--single-stage", default=False, action="store_true")
    argparser.add_argument("--coverage", default=False, action="store_true")
    argparser.add_argument("--time-limit", default=6, type=int, help="run time limit,unit:hours")

    argparser.add_argument("--method", default="scenariofuzz", type=str,
                           help="choose generation method",
                           choices=["scenariofuzz", "avfuzz", "autofuzz", "samota", "behavxplore"])
    return argparser


def main():
    global conf, args, alarm_time, test_scenario, scenario_dict

    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)
    json.dump(time.time(), open(conf.out_dir + "/start_time.json", "w"))

    alarm_time = args.alarm_time
    scene_id = 0
    campaign_cnt = 0

    signal.signal(signal.SIGALRM, handler)
    valuerangemanager = ValueRangeManager()
    sc = scenario_selector.ScenarioSelector()

    time_limit = args.time_limit
    test_start_time = time.time()

    if time_limit == 0:
        raise Exception("time_limit must be greater than 0")

    town_map = "{}".format(conf.town)

    scenario_dict = {
        "ego": {
            "start_point": {
                "x": 232.69346618652344,
                "y": -44.225074768066406,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 91.39320373535156,
                "roll": 0.0
            },
            "end_point": {
                "x": 232.5762481689453,
                "y": -9.31171226501465,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 91.39320373535156,
                "roll": 0.0
            }
        },
        "other_vehicle": [
            {
                "start_point": {
                    "x": 235.6978759765625,
                    "y": -20.406654357910156,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": 91.39320373535156,
                    "roll": 0.0
                },
                "middle_point": [],
                "end_point": None
            },
            {
                "start_point": {
                    "x": 235.1062774658203,
                    "y": -55.402235984802246,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": -179.14419555664062,
                    "roll": 0.0
                },
                "middle_point": [],
                "end_point": None
            }
        ]
    }
    test_scenario = fuzz_utils.TestScenario(conf)

    speed_bounds = [0, 10]
    action_bounds = [-1, 1]
    mutationProb = 0.4  # mutation rate
    crossoverProb = 0.4  # crossover rate
    popSize = conf.max_mutations
    numOfNpc = 2
    numOfTimeSlice = 5
    maxGen = conf.max_cycles

    # -----------------------------------------------------------avfuzz_start-----------------------------------------------------------
    seed_idx = 0
    seed_scenarios = []
    wps = json.load(open(f"./towns_map/Town0{conf.town}.wps.json"))
    for seed_path in os.listdir(conf.seed_dir):
        seed_s = os.path.join(conf.seed_dir, seed_path)
        seed_scenarios.append(json.load(open(seed_s, "r")))
    while True:
        if len(seed_scenarios) > 0:
            seed_scenario = seed_scenarios.pop()
            scenario_dict["ego"]["start_point"] = {
                "x": seed_scenario["sp_x"],
                "y": seed_scenario["sp_y"],
                "z": seed_scenario["sp_z"],
                "roll": seed_scenario["roll"],
                "pitch": seed_scenario["pitch"],
                "yaw": seed_scenario["yaw"],
            }
            scenario_dict["ego"]["end_point"] = {
                "x": seed_scenario["wp_x"],
                "y": seed_scenario["wp_y"],
                "z": seed_scenario["wp_z"],
                "roll": seed_scenario["roll"],
                "pitch": seed_scenario["pitch"],
                "yaw": seed_scenario["wp_yaw"],
            }
            wp1_road_id, wp2_road_id = random.choices(list(wps.keys()), k=2)
            wp1 = random.choice(wps[wp1_road_id])
            wp2 = random.choice(wps[wp2_road_id])
            scenario_dict["other_vehicle"][0] = {
                "start_point": {
                    "x": wp1["location"]["x"],
                    "y": wp1["location"]["y"],
                    "z": wp1["location"]["z"],
                    "pitch": wp1["rotation"]["pitch"],
                    "yaw": wp1["rotation"]["yaw"],
                    "roll": wp1["rotation"]["roll"]
                },
                "middle_point": [],
                "end_point": None
            }
            scenario_dict["other_vehicle"][1] = {
                "start_point": {
                    "x": wp2["location"]["x"],
                    "y": wp2["location"]["y"],
                    "z": wp2["location"]["z"],
                    "pitch": wp2["rotation"]["pitch"],
                    "yaw": wp2["rotation"]["yaw"],
                    "roll": wp2["rotation"]["roll"]
                },
                "middle_point": [],
                "end_point": None
            }
            print(f"load seed scenarios {scenario_dict}")
        else:
            scenario_dict["ego"]["start_point"] = random_select_point(town_map)
            scenario_dict["ego"]["end_point"] = random_select_point(town_map)
        # campaign_cnt 转化为第几次启动GA算法
        conf.time_list = {
            'campaign_start_time': None,
            'cycle_start_time': None,
            'mutation_start_time': None
        }

        campaign_start_time = datetime.datetime.now()
        conf.time_list['campaign_start_time'] = campaign_start_time.strftime('%Y-%m-%d %H:%M:%S')

        GAgenerator = GeneticAlgorithm(speed_bounds, action_bounds, popSize, numOfNpc, numOfTimeSlice, maxGen,
                                       mutationProb, crossoverProb, Chromosome_sf, campaign_cnt)
        print('AVfuzzer is {}-th campaign'.format(campaign_cnt))

        if GAgenerator.ck_path != None:
            ck = open(GAgenerator.ck_path, 'rb')
            GAgenerator.pop = pickle.load(ck)
            ck.close()
        if GAgenerator.isInLis == False:
            GAgenerator.init_pop()

        best, bestIndex = GAgenerator.find_best()
        GAgenerator.g_best = copy.deepcopy(best)

        # cycle_cnt 转化为第几代种子
        for cycle_cnt in range(conf.max_cycles):

            GAgenerator.touched_chs = []
            GAgenerator.cross()
            GAgenerator.mutation(cycle_cnt + 1)
            GAgenerator.select_roulette()

            best, bestIndex = GAgenerator.find_best()
            GAgenerator.bests[cycle_cnt] = best

            ########### Update noprogressCounter #########
            noprogress = False
            ave = 0
            if cycle_cnt >= GAgenerator.lastRestartGen + 5:
                for j in range(cycle_cnt - 5, cycle_cnt):
                    ave += GAgenerator.bests[j].y
                ave /= 5
                if ave >= best.y:
                    GAgenerator.lastRestarGen = cycle_cnt
                    noprogress = True

            if GAgenerator.g_best.y < best.y:  # Record the best fitness score across all generations
                GAgenerator.g_best = copy.deepcopy(best)

            N_generation = GAgenerator.pop
            N_b = GAgenerator.g_best

            # Update the checkpoint of the best scenario so far
            GAgenerator.take_checkpoint(N_b, 'best_scenario.obj', conf.out_dir)

            # Checkpoint this generation
            GAgenerator.take_checkpoint(N_generation, 'last_gen.obj', conf.out_dir)

            # Checkpoint every generation
            now = datetime.datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            GAgenerator.take_checkpoint(N_generation, 'generation-' + str(cycle_cnt) + '-at-' + date_time, conf.out_dir)

            oldCkName = os.path.join(conf.out_dir, 'GaCheckpoints')
            #################### Start the Restart Process ################### 
            if noprogress == True and not GAgenerator.isInLis:
                newPop = generateRestart.generateRestart(oldCkName, 1000,
                                                         (GAgenerator.speed_bounds, GAgenerator.action_bounds),
                                                         Chromosome_sf)
                GAgenerator.pop = copy.deepcopy(newPop)
                GAgenerator.hasRestarted = True
                best, GAgenerator.bestIndex = GAgenerator.find_best()
                GAgenerator.bestYAfterRestart = best.y
                GAgenerator.lastRestartGen = cycle_cnt
            #################### End the Restart Process ################### 

            if os.path.exists(oldCkName) == True:
                prePopPool = generateRestart.getAllCheckpoints(oldCkName)
                simiSum = 0
                for eachChs in GAgenerator.pop:
                    eachSimilarity = generateRestart.getSimularityOfScenarioVsPrevPop(eachChs, prePopPool)
                    simiSum += eachSimilarity

            # Log fitness etc
            f = open('Progress.log', 'a')
            f.write(str(cycle_cnt) + " " + str(best.y) + " " + str(GAgenerator.g_best.y) + " " + str(
                simiSum / float(GAgenerator.pop_size)) + " " + date_time + "\n")
            f.close()

            if best.y > GAgenerator.bestYAfterRestart:
                GAgenerator.bestYAfterRestart = best.y
                if cycle_cnt > (
                        GAgenerator.lastRestartGen + GAgenerator.minLisGen) and GAgenerator.isInLis == False:  # Only allow one level of recursion
                    ################## Start LIS #################
                    # Increase mutation rate a little bit to jump out of local maxima
                    lis = GeneticAlgorithm(GAgenerator.speed_bounds, GAgenerator.action_bounds, (GAgenerator.pm * 1.5),
                                           GAgenerator.pc, GAgenerator.pop_size, GAgenerator.NPC_size,
                                           GAgenerator.time_size, GAgenerator.numOfGenInLis)
                    lis.setLisPop(GAgenerator.g_best)
                    lis.setLisFlag()
                    lisBestChs = lis.ga()
                    if lisBestChs.y > GAgenerator.g_best.y:
                        # Let's replace this
                        GAgenerator.pop[bestIndex] = copy.deepcopy(lisBestChs)

        campaign_cnt += 1


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
