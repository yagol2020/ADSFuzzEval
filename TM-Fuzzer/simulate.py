import coverage

cov = coverage.Coverage()
cov.start()
########################
import copyreg
import fcntl
import glob
import logging

# Python packages
import os
import pdb
import random
import select
import shutil
import subprocess
import sys
import threading
import signal
import time
import math
import traceback

import networkx as nx
import numpy as np
import pygame

from oracle import AllOracle
from npc import NPC
import constants as c
from utils import (
    quaternion_from_euler,
    set_traffic_lights_state,
    get_angle_between_vectors,
    set_autopilot,
    delete_npc,
    check_autoware_status,
    mark_npc,
    timeout_handler,
)

import carla
from agents.navigation.behavior_agent import BehaviorAgent
import dill

if os.environ["ADS"] == "interfuser":
    from InterFuser.team_code.interfuser_agent import InterfuserAgent
    from InterFuser.leaderboard.utils.route_manipulation import (
        interpolate_trajectory as interpolate_trajectory_interfuser,
    )
    from InterFuser.leaderboard.autoagents.agent_wrapper import (
        AgentWrapper as AgentWrapper_interfuser,
    )
    from InterFuser.srunner.scenariomanager.timer import (
        GameTime as GameTime_interfuser,
    )
    from InterFuser.srunner.scenariomanager.carla_data_provider import (
        CarlaDataProvider as CarlaDataProvider_interfuser,
    )
if os.environ["ADS"] == "lmdrive":
    from LMDrive.team_code.lmdriver_agent import LMDriveAgent
    from LMDrive.leaderboard.autoagents.agent_wrapper import (
        AgentWrapper as AgentWrapper_lmdriver,
    )
    from LMDrive.srunner.scenariomanager.carla_data_provider import (
        CarlaDataProvider as CarlaDataProvider_lmdriver,
    )
    from LMDrive.srunner.scenariomanager.timer import (
        GameTime as GameTime_lmdriver,
    )
    from LMDrive.leaderboard.utils.route_manipulation import (
        interpolate_trajectory as interpolate_trajectory_lmdriver,
    )
if os.environ["ADS"] == "autoware_0903":
    print("Activate ENV ADS: Autoware 0903")
    import autoware_2409_utils
    from simulation.sensors import (
        GnssSensor,
        IMUSensor,
        LidarSensor,
        RgbCamera,
    )

    import threading
    from datetime import datetime


def record_min_distance(npc_vehicles, player_loc, state):
    min_dist = state.min_dist
    closest_car = None
    for npc_vehicle in npc_vehicles:
        distance = npc_vehicle.get_location().distance(player_loc)
        if distance < min_dist:
            min_dist = distance
            closest_car = npc_vehicle
    if min_dist < state.min_dist:
        state.min_dist = min_dist
        state.closest_car = closest_car


def simulate(conf, state, exec_state, sp, wp, weather_dict, npc_list):
    carla_port_docker = os.environ.get("CARLA_PORT_DOCKER", "2000")
    if conf.agent_type == c.AUTOWARE_0903:
        carla_v = "0.9.14"
    else:
        carla_v = "0.9.13"
    subprocess.Popen(
        ["./run_carla_v2.sh", "-p", str(carla_port_docker), "-v", carla_v, "-d"]
    )
    time.sleep(10)
    try:
        client = carla.Client("localhost", int(carla_port_docker))
        client.set_timeout(10)
        print("Load Map in simulate.py",f"Town0{conf.town}")
        world = client.load_world(f"Town0{conf.town}")
    except Exception as carla_connect_error:
        print(carla_connect_error)
        subprocess.Popen(
            ["./run_carla_v2.sh", "-p", str(carla_port_docker), "-v", carla_v, "-d"]
        )
        time.sleep(10)
        return simulate(conf, state, exec_state, sp, wp, weather_dict, npc_list)
    if os.path.exists("/tmp/fuzzerdata-tmfuzz"):
        shutil.rmtree("/tmp/fuzzerdata-tmfuzz")
    os.mkdir("/tmp/fuzzerdata-tmfuzz")
    #####################
    state.location_recorder = []
    state.running_red_light_details = []
    state.stuck_details = []
    state.speeding_details = []
    state.collision_details = []
    state.laneinvasion_details = []
    state.start_simulate_time = time.time()
    #####################
    all_oracle = None
    #####################

    retval = 0
    wait_until_end = 0
    max_wheels_for_non_motorized = 2
    carla_error = False
    state.min_dist = 99999
    player_loc = None
    time_start = time.time()
    npc_now = []
    agents_now = []
    sensors = []
    npc_vehicles = []
    npc_walkers = []
    trace_graph = []
    trace_graph_important = []
    nearby_dict = []

    # for autoware
    frame_gap = 0
    autoware_last_frames = 0
    autoware_stuck = 0

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE  # FPS
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.tick()
    if conf.agent_type == c.INTERFUSER:
        CarlaDataProvider_interfuser.set_client(client)
        CarlaDataProvider_interfuser.set_world(world)
    elif conf.agent_type == c.LMDRIVE:
        CarlaDataProvider_lmdriver.cleanup()
        CarlaDataProvider_lmdriver.set_client(client)
        CarlaDataProvider_lmdriver.set_world(world)
        GameTime_lmdriver.restart()
        print("LMDrive: initial data Provider and gametime")
    elif conf.agent_type == c.AUTOWARE_0903:
        autoware_2409_utils.stop_autoware()
        autoware_2409_utils.stop_bridge()
    for n in npc_list:
        if n.npc_bp != None:
            n.npc_bp = world.get_blueprint_library().find(n.npc_bp)
            print(n.npc_bp)
    goal_loc = wp.location
    goal_rot = wp.rotation

    trace_dict = {}
    valid_frames = 0
    all_frame = 0
    try:

        # initialize the simulation and the ego vehicle

        (
            add_car_frame,
            blueprint_library,
            clock,
            player_bp,
            town_map,
            vehicle_bp_library,
        ) = simulate_initialize(client, conf, weather_dict, world)
        #######################################
        # for autoware
        autoware_container = None
        autoware_state_monitor = None
        town = world.get_map()
        player = None
        print(sp,wp)
        if conf.agent_type == c.BEHAVIOR:
            player = world.try_spawn_actor(player_bp, sp)
            ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
            ego.set_instance(player)
            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)
            agent = BehaviorAgent(player, behavior="cautious")
            agent.set_destination(
                start_location=sp.location,
                end_location=wp.location,
            )
            agents_now.append((agent, player, ego))
            print("[+] spawned cautious BehaviorAgent")
        elif conf.agent_type == c.INTERFUSER:
            player = world.try_spawn_actor(player_bp, sp)
            ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
            ego.set_instance(player)
            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)
            agent = InterfuserAgent("InterFuser/team_code/interfuser_config.py")
            trajectory = [transform_2_location(sp), transform_2_location(wp)]
            gps_route, route = interpolate_trajectory_interfuser(world, trajectory)
            agent.set_global_plan(gps_route, route)
            agent_wrapper = AgentWrapper_interfuser(agent)
            agent_wrapper.setup_sensors(player)
            agents_now.append((agent, player, ego))
            print(
                "[+] spawned cautious InterFuser: From %s to %s"
                % (sp.location, wp.location)
            )
        elif conf.agent_type == c.LMDRIVE:
            player = CarlaDataProvider_lmdriver.request_new_actor(
                player_bp.id, sp, rolename="hero"
            )
            ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
            ego.set_instance(player)
            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)
            agent = LMDriveAgent("LMDrive/team_code/lmdriver_config.py")
            agent.town_id = town.replace("Carla/Maps/", "")
            agent.sampled_scenarios = []
            agent.scenario_cofing_name = str(random.randint(1, 2025))
            ego_sp = carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z)
            ego_dp = carla.Location(x=wp.location.x, y=wp.location.y, z=wp.location.z)
            trajectory = [ego_sp, ego_dp]
            gps_route, route = interpolate_trajectory_lmdriver(world, trajectory)
            agent.set_global_plan(gps_route, route)
            agent_wrapper = AgentWrapper_lmdriver(agent)
            agent_wrapper.setup_sensors(player)
            agents_now.append((agent, player, ego))
            print("[+] spawned LMDrive Agent")

        elif conf.agent_type == c.AUTOWARE_0903:
            player_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
            player_bp.set_attribute("role_name", "autoware_v1")
            player = world.try_spawn_actor(player_bp, sp)
            try:
                physics_control = player.get_physics_control()
                physics_control.use_sweep_wheel_collision = True
                player.apply_physics_control(physics_control)
            except Exception as e:
                print(e)
            ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
            ego.set_instance(player)
            _gnss_sensor = GnssSensor(player, sensor_name="ublox")
            _imu_sensor = IMUSensor(player, sensor_name="tamagawa")
            _lidar_sensor = LidarSensor(player, sensor_name="top")
            _rgb_camera = RgbCamera(player, sensor_name="traffic_light")
            world.tick()
            max_wait = 60
            autoware_2409_utils.start_bridge()  # start bridge in backgroun
            time.sleep(10)
            autoware_2409_utils.start_autoware()  # open autoware
            while True:
                if autoware_2409_utils.autoware_start_ready:
                    break
                elif max_wait < 0:
                    print("Autoware start fail!")
                    retval = -1
                    raise Exception("Autoware start fail!")
                else:
                    time.sleep(1)
                    print(f"Waiting for Autoware ready... [{max_wait}s]")
                    max_wait -= 1
            autoware_2409_utils.send_goal_pose(wp)
            time.sleep(5)  # wait for goal ready
            autoware_2409_utils.restart_ros_daemon()
            autoware_state_monitor = threading.Thread(
                target=autoware_2409_utils.monitor_routing_state
            )
            autoware_state_monitor.daemon = True
            autoware_state_monitor.start()
            max_wait = 60
            while True:
                if autoware_2409_utils.goal_ready == 1:  # good set
                    break
                elif autoware_2409_utils.goal_ready == 2:  # error set
                    print("Goal Set is Error")
                    retval = -1
                    raise Exception("Autoware Goad Set Error")
                elif autoware_2409_utils.goal_ready == 0:
                    print(f"Waiting Autoware Set goal...[{max_wait}]")
                    time.sleep(1)
                    max_wait -= 1
                if max_wait < 0:
                    print("Goooal Set fail")
                    retval = -1
                    raise Exception("Goooal Set fail")
            agents_now.append(("Autoware_0903", player, ego))
            print("[+] spawned Autowre 0903 Agent")

        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "800")
        rgb_camera_bp.set_attribute("image_size_y", "600")
        rgb_camera_bp.set_attribute("fov", "105")

        camera_tf = carla.Transform(carla.Location(z=1.8))
        camera_front = world.spawn_actor(
            rgb_camera_bp,
            camera_tf,
            attach_to=player,
            attachment_type=carla.AttachmentType.Rigid,
        )

        camera_front.listen(lambda image: _on_front_camera_capture(image, state))
        sensors.append(camera_front)
        camera_tf2 = carla.Transform(
            carla.Location(z=50.0), carla.Rotation(pitch=-90.0)
        )
        camera_top = world.spawn_actor(
            rgb_camera_bp,
            camera_tf2,
            attach_to=player,
            attachment_type=carla.AttachmentType.Rigid,
        )

        camera_top.listen(lambda image: _on_top_camera_capture(image, state))
        sensors.append(camera_top)

        ego.attach_collision(world, sensors, state)
        if conf.check_dict["lane"]:
            ego.attach_lane_invasion(world, sensors, state)
        # get vehicle's maximum steering angle
        physics_control = player.get_physics_control()
        max_steer_angle = 0
        for wheel in physics_control.wheels:
            if wheel.max_steer_angle > max_steer_angle:
                max_steer_angle = wheel.max_steer_angle
        if conf.agent_type != c.AUTOWARE_0903:
            world.tick()  # sync with simulator
            print("Tick!")
        #######################################
        # autoware_container, ego, player, max_steer_angle, autoware_state_monitor = (
        #     ego_initialize(
        #         agents_now,
        #         exec_state,
        #         blueprint_library,
        #         conf,
        #         player_bp,
        #         sensors,
        #         sp,
        #         state,
        #         world,
        #         wp,
        #     )
        # )
        trace_dict[player.id] = []

        # SIMULATION LOOP FOR AUTOWARE and BasicAgent
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGSEGV, state.sig_handler)
        signal.signal(signal.SIGABRT, state.sig_handler)

        try:

            found_frame = -999
            snapshot0 = world.get_snapshot()
            first_frame_id = snapshot0.frame
            first_sim_time = snapshot0.timestamp.elapsed_seconds
            if conf.agent_type == c.INTERFUSER:
                GameTime_interfuser.on_carla_tick(snapshot0.timestamp)
            elif conf.agent_type == c.LMDRIVE:
                GameTime_lmdriver.on_carla_tick(snapshot0.timestamp)
            last_frame_id = first_frame_id
            state.first_frame_id = first_frame_id
            state.sim_start_time = snapshot0.timestamp.platform_timestamp
            state.num_frames = 0
            state.elapsed_time = 0
            frame_speed_lim_changed = 0
            s_started = False
            # while True:
            #     time.sleep(1)

            if conf.agent_type == c.AUTOWARE_0903:
                if (
                    not autoware_2409_utils.change_to_autonomous_mode()
                ):  # Set to Autopilot
                    print("Set autopilot Error! Exit")
                    retval = -1
                    raise Exception("Autoware Set Autopilot Error")
            # actual monitoring of the driving simulation begins here
            print("START DRIVING: {} {}".format(first_frame_id, first_sim_time))
            # mark goal position
            # if conf.debug:
            #     world.debug.draw_box(
            #         box=carla.BoundingBox(goal_loc, carla.Vector3D(0.2, 0.2, 1.0)),
            #         rotation=goal_rot,
            #         life_time=0,
            #         thickness=1.0,
            #         color=carla.Color(r=0, g=255, b=0),
            #     )
            # simulate start here
            state.end = False
            time_start = time.time()
            print("Start loop")
            all_oracle = AllOracle(player, world, conf.out_dir)
            while time.time() - time_start < 10 * 60:
                # world tick
                if (
                    conf.agent_type == c.BEHAVIOR
                    or conf.agent_type == c.INTERFUSER
                    or conf.agent_type == c.LMDRIVE
                ):
                    world.tick()
                # Use sampling frequency of FPS  for precision
                clock.tick(c.FRAME_RATE)
                # Get frame info
                snapshot = world.get_snapshot()
                cur_frame_id = snapshot.frame
                cur_sim_time = snapshot.timestamp.elapsed_seconds
                if conf.agent_type == c.INTERFUSER:
                    GameTime_interfuser.on_carla_tick(snapshot.timestamp)
                elif conf.agent_type == c.LMDRIVE:
                    GameTime_lmdriver.on_carla_tick(snapshot.timestamp)
                if cur_frame_id <= last_frame_id:
                    # skip if we got the same frame data as last
                    continue
                last_frame_id = cur_frame_id  # update last
                state.num_frames = cur_frame_id - first_frame_id
                state.elapsed_time = cur_sim_time - first_sim_time

                (
                    frame_speed_lim_changed,
                    player_lane_id,
                    player_loc,
                    player_road_id,
                    player_rot,
                    speed,
                    speed_limit,
                    vel,
                    player_tras,
                ) = get_player_info(
                    cur_frame_id, goal_loc, player, sp, state, town_map, conf
                )

                state.location_recorder.append((player_loc.x, player_loc.y))
                carla_spectator = world.get_spectator()
                camera_location = player_loc + carla.Location(z=20)
                camera_rotation = carla.Rotation(pitch=-90)
                carla_spectator.set_transform(
                    carla.Transform(camera_location, camera_rotation)
                )
                # Check destination
                break_flag, retval, autoware_stuck, s_started, wq = check_destination(
                    npc_vehicles,
                    npc_now,
                    agents_now,
                    autoware_stuck,
                    conf,
                    goal_loc,
                    goal_rot,
                    player_loc,
                    exec_state.proc_state,
                    retval,
                    s_started,
                    sensors,
                    speed,
                    state,
                    world,
                )
                # record the min distance between every two npcs
                record_min_distance(npc_vehicles, player_loc, state)
                # mark useless vehicles for any frame
                mark_useless_npc(
                    npc_now,
                    conf,
                    player_lane_id,
                    player_loc,
                    player_rot,
                    player_road_id,
                    exec_state.G,
                    town_map,
                )

                # add old vehicles for any frame
                found_frame = add_old_npc(
                    npc_list,
                    npc_vehicles,
                    npc_now,
                    agents_now,
                    conf,
                    found_frame,
                    max_wheels_for_non_motorized,
                    player_loc,
                    sensors,
                    state,
                    vel,
                    world,
                    wp,
                )
                # add a new npc per 1s here
                # because of autoware's strange frame, we should use a interesting method
                found_frame, autoware_last_frames, frame_gap = add_new_car(
                    npc_list,
                    npc_vehicles,
                    npc_now,
                    add_car_frame,
                    agents_now,
                    autoware_last_frames,
                    conf,
                    ego,
                    found_frame,
                    frame_gap,
                    goal_loc,
                    goal_rot,
                    max_wheels_for_non_motorized,
                    player_lane_id,
                    player_loc,
                    player_road_id,
                    sensors,
                    state,
                    town_map,
                    vehicle_bp_library,
                    world,
                    wp,
                    exec_state.G,
                )
                control_npc(agents_now, speed_limit)
                all_oracle.update(snapshot, goal_loc, wq)
                # delete vehicles which life is end
                for npc in npc_list:
                    if npc.instance is not None:
                        if npc.death_time == 0:
                            delete_npc(npc, npc_vehicles, sensors, agents_now, npc_now)
                        elif npc.death_time > 0:
                            npc.death_time -= 1

                # record track of every npc_vehicle
                if break_flag:
                    break
                if wait_until_end == 0:
                    valid_frames = nearby_record(
                        state,
                        npc_vehicles,
                        player,
                        trace_dict,
                        player_loc,
                        town_map,
                        valid_frames,
                        exec_state.G,
                    )
                    all_frame += 1
                    retval, wait_until_end = check_violation(
                        conf,
                        cur_frame_id,
                        frame_speed_lim_changed,
                        retval,
                        speed,
                        speed_limit,
                        state,
                        wait_until_end,
                        snapshot=snapshot,
                        player_transform=player_tras,
                    )
                else:
                    wait_until_end += 1
                if wait_until_end > 6:
                    break
            # find the biggest weight of npc-list
            nearby_dict, trace_graph_important = record_trace(
                npc_vehicles,
                exec_state,
                player,
                player_loc,
                state,
                town_map,
                trace_dict,
                trace_graph,
                trace_graph_important,
            )
            state.trace_graph_important = trace_graph_important
            state.nearby_dict = nearby_dict
        except KeyboardInterrupt:
            print("quitting")
            retval = 128
        # jump to finally
        return
    except Exception:
        # update states
        # state.num_frames = frame_id - frame_0
        # state.elapsed_time = time.time() - start_time
        print("[-] Runtime error:")
        traceback.print_exc()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("   (line #{0}) {1}".format(exc_tb.tb_lineno, exc_type))
        # retval = -1
    finally:
        if conf.agent_type == c.AUTOWARE_0903:
            autoware_2409_utils.monitor_routing_event.set()
            autoware_2409_utils.stop_autoware()
            autoware_2409_utils.stop_bridge()
        if all_oracle:
            all_oracle.save_2_file()
            cov.stop()
            cov.save()
            cov.xml_report(
                outfile=os.path.join(
                    conf.out_dir,
                    "oracles",
                    str(all_oracle.start_sim_time),
                    "coverage.xml",
                )
            )
            if conf.agent_type == c.AUTOWARE_0903:
                autoware_dt = datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
                autoware_2409_utils.lcov_once(label=str(autoware_dt))
                if os.path.exists(
                    os.path.expanduser(
                        f"~/autoware_carla_launch/lcov_coverage/html_{autoware_dt}"
                    )
                ):
                    shutil.copytree(
                        os.path.expanduser(
                            f"/home/yy/autoware_carla_launch/lcov_coverage/html_{autoware_dt}"
                        ),
                        all_oracle.report_dir + f"/html_{autoware_dt}",
                    )
        signal.alarm(0)
        all_time = time.time() - time_start
        FPS = all_frame / all_time
        valid_time = (valid_frames / FPS) if FPS != 0 else 0
        logging.info("crashed:%s", state.crashed)
        logging.info("nearby_car:%s", len(nearby_dict))
        logging.info("valid_time/time: %s/%s", valid_time, all_time)
        logging.info("distance:%s", state.distance)
        logging.info("FPS:%s", FPS)
        state.end = True
        state.stop_simulate_time = time.time()
        if exec_state.proc_state:
            exec_state.proc_state.terminate()
            exec_state.proc_state.wait()
            exec_state.proc_state.stdout.close()
            exec_state.proc_state.stderr.close()

        # save video in output_dir
        if (
            conf.agent_type == c.BEHAVIOR
            or conf.agent_type == c.INTERFUSER
            or conf.agent_type == c.LMDRIVE
            or conf.agent_type == c.AUTOWARE_0903
        ):
            save_behavior_video(carla_error, state)
        # Finalize simulation
        if retval == -1:
            print("[debug] exit because of Runtime error")
            return retval, npc_list, state
        if conf.debug:
            print("[debug] reload")
        for npc in npc_now:
            npc.instance = None
        for npc in npc_list:
            npc.fresh = True
        # client.reload_world()
        if retval == 128:
            print("[debug] exit by user requests")
            return retval, npc_list, state
        return retval, npc_list, state


def record_trace(
    npc_vehicles,
    exec_state,
    player,
    player_loc,
    state,
    town_map,
    trace_dict,
    trace_graph,
    trace_graph_important,
):
    nearby_dict = {}
    for vehicle_id in trace_dict:
        if player.id == vehicle_id:
            continue
        nearby_dict[vehicle_id] = len(trace_dict[vehicle_id])
    # record nearby cars when test is end
    if town_map:
        player_waypoint = town_map.get_waypoint(
            player_loc, project_to_road=True, lane_type=carla.libcarla.LaneType.Driving
        )
    else:
        return [], []
    for npc_vehicle in npc_vehicles:
        if state.crashed:
            if state.collision_to == npc_vehicle.id:
                continue
        waypoint = town_map.get_waypoint(
            npc_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Driving,
        )
        if check_topo(player_waypoint, waypoint, exec_state.G):
            # trim and thin the trace,return trace at last 5 seconds
            try:
                trace = trace_dict[npc_vehicle.id]
            except KeyError:
                continue
            # trace = trace_thin(trace, 5, 25)
            trace_graph.append(trace)
    trace_graph_important.append(trace_dict[player.id])
    if state.crashed:
        # if collied to a car
        if trace_dict.keys().__contains__(state.collision_to):
            trace_graph_important.append(trace_dict[state.collision_to])
    else:
        # todo:better
        for trace in trace_dict:
            trace_graph_important.append(trace_dict[trace])
            break

    # Do not change the speed
    ego_start_loc = (trace_graph_important[0][0][0], trace_graph_important[0][0][1], 0)
    for i in range(len(trace_graph_important)):
        # save at most 250 points
        if len(trace_graph_important[i]) > 250:
            trace_graph_important[i] = trace_graph_important[i][-250:]
        normalize_points(trace_graph_important[i], ego_start_loc)
    # change trace_graph_important from list to ndarray
    trace_graph_important = np.array(trace_graph_important)
    return nearby_dict, trace_graph_important


def normalize_points(points, start_point):
    origin_point = np.array(start_point)
    points = np.array(points)
    for i in range(len(points)):
        points[i] = points[i] - origin_point


def nearby_record(
    state, npc_vehicles, player, trace_dict, player_loc, town_map, valid_frames, G
):
    vel = player.get_velocity()
    speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    player_waypoint = town_map.get_waypoint(
        player_loc, project_to_road=True, lane_type=carla.libcarla.LaneType.Driving
    )
    trace_dict[player.id].append((player_loc.x, player_loc.y, speed))
    has_nearby = False
    for npc_vehicle in npc_vehicles:
        if npc_vehicle.id not in trace_dict:
            trace_dict[npc_vehicle.id] = []
        waypoint = town_map.get_waypoint(
            npc_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Driving,
        )
        is_nearby = check_topo(player_waypoint, waypoint, G)
        npc_vehicle_speed = 3.6 * math.sqrt(
            npc_vehicle.get_velocity().x ** 2 + npc_vehicle.get_velocity().y ** 2
        )
        not_stuck = (npc_vehicle_speed > 0.5) or (npc_vehicle_speed > 0.5)
        if is_nearby and not_stuck:
            trace_dict[npc_vehicle.id].append(
                (
                    npc_vehicle.get_location().x,
                    npc_vehicle.get_location().y,
                    npc_vehicle_speed,
                )
            )
            has_nearby = True
    if has_nearby and state.stuck_duration < 15 * c.FRAME_RATE:
        valid_frames += 1
    return valid_frames


def control_npc(agents_now, speed_limit):
    for agent_tuple in agents_now:
        if os.environ["ADS"] == "interfuser" and isinstance(
            agent_tuple[0], InterfuserAgent
        ):
            control = agent_tuple[0]()
            agent_vehicle = agent_tuple[1]
            agent_npc = agent_tuple[2]
            agent_vehicle.apply_control(control)
        elif os.environ["ADS"] == "lmdrive" and isinstance(
            agent_tuple[0], LMDriveAgent
        ):
            control = agent_tuple[0]()
            agent_vehicle = agent_tuple[1]
            agent_npc = agent_tuple[2]
            agent_vehicle.apply_control(control)
        elif os.environ["ADS"] == "autoware_0903":
            agent_vehicle = agent_tuple[1]
            agent_npc = agent_tuple[2]
        else:
            # todo:rewrite here
            agent = agent_tuple[0]
            agent_vehicle = agent_tuple[1]
            agent_npc = agent_tuple[2]
            agent._update_information()
            agent.get_local_planner().set_speed(speed_limit)
            lp = agent.get_local_planner()
            if len(lp._waypoints_queue) != 0:
                control = agent.run_step()
                # that guy who has the agent
                agent_vehicle.apply_control(control)

        vel = agent_vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        # Check inactivity
        if speed < 1:  # km/h
            agent_npc.stuck_duration += 1
        else:
            agent_npc.stuck_duration = 0


def add_new_car(
    npc_list,
    npc_vehicles,
    npc_now,
    add_car_frame,
    agents_now,
    autoware_last_frames,
    conf,
    ego,
    found_frame,
    frame_gap,
    goal_loc,
    goal_rot,
    max_wheels_for_non_motorized,
    player_lane_id,
    player_loc,
    player_road_id,
    sensors,
    state,
    town_map,
    vehicle_bp_library,
    world,
    wp,
    G,
):
    if conf.agent_type == c.AUTOWARE:
        frame_gap = frame_gap + state.num_frames - autoware_last_frames
        autoware_last_frames = state.num_frames
        add_flag = frame_gap > add_car_frame - 1
        if add_flag:
            frame_gap = frame_gap - add_car_frame
    else:
        add_flag = state.num_frames % add_car_frame == add_car_frame - 1
    if add_flag:
        # try to spawn a test linear car to see if the simulation is still running
        # a choose npc from npc_list first
        # delete background vehicle which is too far
        if abs(state.num_frames - found_frame) > add_car_frame:
            add_type = random.randint(1, 100)  # this value controls the type of npc
            npc_vehicle = None
            new_npc = None
            repeat_times = 0
            while npc_vehicle is None:
                repeat_times += 1
                # stuck too long
                if repeat_times > 100 or state.stuck_duration > 100:
                    # add a fake npc
                    new_npc = NPC(
                        npc_type=None,
                        spawn_point=None,
                        speed=None,
                        npc_id=len(npc_list),
                        ego_loc=player_loc,
                    )
                    new_npc.instance = None
                    npc_list.append(new_npc)
                    new_npc.fresh = False
                    break
                x = random.uniform(-50, 50)
                y = random.uniform(-50, 50)
                # 1.don't add vehicles that are physically too far away
                if x**2 + y**2 > 50**2:
                    repeat_times -= 1
                    continue
                # 2.don't add vehicles that are topologically too far away
                location = carla.Location(
                    x=player_loc.x + x, y=player_loc.y + y, z=player_loc.z
                )
                waypoint = town_map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=carla.libcarla.LaneType.Driving,
                )
                neighbors_A = nx.single_source_shortest_path_length(
                    G, source=(player_road_id, player_lane_id), cutoff=conf.topo_k
                )
                neighbors_A[(player_road_id, player_lane_id)] = 0
                neighbors_B = nx.single_source_shortest_path_length(
                    G, source=(waypoint.road_id, waypoint.lane_id), cutoff=conf.topo_k
                )
                neighbors_B[(waypoint.road_id, waypoint.lane_id)] = 0
                if not any(
                    node in neighbors_A and node in neighbors_B for node in G.nodes()
                ):
                    repeat_times -= 1
                    continue
                # 3.we don't want to add bg car in junction or near it
                # because it may cause a red light problem
                if waypoint.is_junction:
                    continue
                # if waypoint.is_junction or waypoint.next(30 * 3 / 3.6)[-1].is_junction:
                #     continue
                temp_flag = False
                # 4. don't add a vehicle in the same lane
                for other_npc in npc_list:
                    if other_npc.instance is not None:
                        other_npc_waypoint = other_npc.get_waypoint(town_map)
                        if (other_npc_waypoint.lane_id == waypoint.lane_id) & (
                            other_npc_waypoint.road_id == waypoint.road_id
                        ):
                            temp_flag = True
                            break
                if player_loc.distance(waypoint.transform.location) < 20:
                    if (player_lane_id == waypoint.lane_id) & (
                        player_road_id == waypoint.road_id
                    ):
                        temp_flag = True
                if temp_flag:
                    continue
                # we don't want to add an immobile bg car at lane that can't change lane
                if waypoint.lane_change == carla.LaneChange.NONE:
                    if add_type <= conf.immobile_percentage:
                        repeat_times -= 1
                        continue
                road_direction = waypoint.transform.rotation.get_forward_vector()
                road_direction_x = road_direction.x
                road_direction_y = road_direction.y
                roll = math.atan2(road_direction_y, road_direction_x)
                roll_degrees = math.degrees(roll)
                npc_spawn_point = carla.Transform(
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z + 0.1,
                    ),
                    carla.Rotation(pitch=0, yaw=roll_degrees, roll=0),
                )
                # random choose a car bp from vehicle_bp_library
                npc_bp = random.choice(vehicle_bp_library)

                if add_type <= conf.immobile_percentage:
                    # add a immobile car
                    bg_speed = 0
                    new_npc = NPC(
                        npc_type=c.VEHICLE,
                        spawn_point=npc_spawn_point,
                        speed=bg_speed,
                        npc_id=len(npc_list),
                        ego_loc=player_loc,
                        npc_bp=npc_bp,
                        spawn_stuck_frame=state.stuck_duration,
                    )
                else:
                    bg_speed = random.uniform(0 / 3.6, 20 / 3.6)
                    new_npc = NPC(
                        npc_type=c.VEHICLE,
                        spawn_point=npc_spawn_point,
                        speed=bg_speed,
                        npc_id=len(npc_list),
                        ego_loc=player_loc,
                        npc_bp=npc_bp,
                        spawn_stuck_frame=state.stuck_duration,
                    )
                # do safe check
                flag = True
                for npc in npc_now:
                    if not new_npc.safe_check(npc):
                        flag = False
                        break
                if not new_npc.safe_check(ego):
                    flag = False
                if flag:
                    npc_vehicle = world.try_spawn_actor(new_npc.npc_bp, npc_spawn_point)
                else:
                    continue
            if npc_vehicle is not None:
                spawn_npc(
                    new_npc,
                    npc_vehicle,
                    npc_vehicles,
                    npc_now,
                    agents_now,
                    conf,
                    max_wheels_for_non_motorized,
                    road_direction,
                    sensors,
                    state,
                    world,
                    wp,
                )
                npc_list.append(new_npc)
        found_frame = False
    return found_frame, autoware_last_frames, frame_gap


def add_old_npc(
    npc_list,
    npc_vehicles,
    npc_now,
    agents_now,
    conf,
    found_frame,
    max_wheels_for_non_motorized,
    player_loc,
    sensors,
    state,
    vel,
    world,
    wp,
):
    for npc in npc_list:
        if npc.fresh & (npc.ego_loc.distance(player_loc) < 1.5):
            found_frame = state.num_frames
            # check if this npc is good to spawn
            v1 = carla.Vector2D(
                npc.ego_loc.x - player_loc.x, npc.ego_loc.y - player_loc.y
            )
            v2 = vel
            if state.stuck_duration != 0:
                # if ego is stuck,check stuck duration
                if npc.spawn_stuck_frame != state.stuck_duration:
                    continue
            angle = get_angle_between_vectors(v1, v2)
            if angle < 90 and angle != 0:
                # the better time will come later
                continue
            # check if this npc is not exist
            if npc.npc_type is None:
                npc.fresh = False
                break
            npc_vehicle = world.try_spawn_actor(npc.npc_bp, npc.spawn_point)
            if npc_vehicle is not None:
                npc_spawn_rotation = npc.spawn_point.rotation
                roll_degrees = npc_spawn_rotation.yaw
                roll = math.radians(roll_degrees)
                road_direction_x = math.cos(roll)
                road_direction_y = math.sin(roll)
                road_direction = carla.Vector3D(road_direction_x, road_direction_y, 0.0)
                spawn_npc(
                    npc,
                    npc_vehicle,
                    npc_vehicles,
                    npc_now,
                    agents_now,
                    conf,
                    max_wheels_for_non_motorized,
                    road_direction,
                    sensors,
                    state,
                    world,
                    wp,
                )
                continue
    return found_frame


def is_in_front(yaw_degrees, car1_position, car2_position):
    yaw_radians = math.radians(yaw_degrees)
    direction_vector = (math.cos(yaw_radians), math.sin(yaw_radians))

    vector_12 = (car2_position.x - car1_position.x, car2_position.y - car1_position.y)
    angle_12 = math.atan2(vector_12[1], vector_12[0])

    angle_diff = math.degrees(angle_12 - yaw_radians)
    angle_diff = (angle_diff + 180) % 360 - 180  # Normalize to [-180, 180]

    return -30 <= angle_diff <= 30


def mark_useless_npc(
    npc_now, conf, player_lane_id, player_loc, player_rot, player_road_id, G, town_map
):
    for npc in npc_now:
        if npc.death_time != -1:
            # (npc.death_time == -1) means it is alive
            continue
        # if is_in_front(player_rot.yaw, player_loc, npc.instance.get_location()):
        #     print("npc in front")
        #     continue
        vehicle = npc.instance
        vehicle_waypoint = town_map.get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Driving,
        )
        vehicle_lane_id = vehicle_waypoint.lane_id
        vehicle_road_id = vehicle_waypoint.road_id
        # 1. Delete vehicles that are physically too far away
        if vehicle.get_location().distance(player_loc) > 100 * math.sqrt(2):
            if is_in_front(player_rot.yaw, player_loc, npc.instance.get_location()):
                continue
            mark_npc(npc, 0)
            # delete_npc(npc, npc_vehicles, sensors, agents_now, npc_now)
            continue
        # 2. Delete vehicles that are topologically too far away
        neighbors_A = nx.single_source_shortest_path_length(
            G, source=(player_road_id, player_lane_id), cutoff=conf.topo_k + 1
        )
        neighbors_A[(player_road_id, player_lane_id)] = 0
        neighbors_B = nx.single_source_shortest_path_length(
            G, source=(vehicle_road_id, vehicle_lane_id), cutoff=conf.topo_k + 1
        )
        neighbors_B[(vehicle_road_id, vehicle_lane_id)] = 0
        if not any(node in neighbors_A and node in neighbors_B for node in G.nodes()):
            if is_in_front(player_rot.yaw, player_loc, npc.instance.get_location()):
                continue
            mark_npc(npc, 1 * c.FRAME_RATE)
        # 3. Delete vehicles that stuck too long
        # Update: According to the reviewer's opinion, the weight is increased here to ensure scene coverage.
        if npc.stuck_duration > (conf.timeout * c.FRAME_RATE / 10):
            # let stuck car go away
            npc_rot = npc.instance.get_transform().rotation
            yaw_radians = math.radians(npc_rot.yaw)
            direction_vector = (math.cos(yaw_radians), math.sin(yaw_radians))
            velocity_magnitude = 5
            velocity = carla.Vector3D(
                velocity_magnitude * direction_vector[0],
                velocity_magnitude * direction_vector[1],
                0,
            )
            npc.instance.set_target_velocity(velocity)
            mark_npc(npc, 5 * c.FRAME_RATE)


def get_player_info(cur_frame_id, goal_loc, player, sp, state, town_map, conf=None):
    # Get player info
    frame_speed_lim_changed = 0
    player_transform = player.get_transform()
    player_loc = player_transform.location
    player_rot = player_transform.rotation
    player_waypoint = town_map.get_waypoint(
        player_loc, project_to_road=True, lane_type=carla.libcarla.LaneType.Driving
    )
    player_lane_id = player_waypoint.lane_id
    player_road_id = player_waypoint.road_id
    vel = player.get_velocity()
    speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    speed_limit = player.get_speed_limit()
    try:
        last_speed_limit = state.speed_lim[-1]
    except:
        last_speed_limit = 0
    if speed_limit != last_speed_limit:
        frame_speed_lim_changed = cur_frame_id
    state.speed.append(speed)
    state.speed_lim.append(speed_limit)
    state.distance += speed / 3.6 / c.FRAME_RATE
    if conf.debug:
        print(
            "[debug] (%.2f,%.2f)>(%.2f,%.2f)>(%.2f,%.2f) %.2f m left, %.2f/%d km/h   \r"
            % (
                sp.location.x,
                sp.location.y,
                player_loc.x,
                player_loc.y,
                goal_loc.x,
                goal_loc.y,
                player_loc.distance(goal_loc),
                speed,
                speed_limit,
            ),
            end="",
        )
    if player.is_at_traffic_light():
        traffic_light = player.get_traffic_light()
        if traffic_light.get_state() == carla.TrafficLightState.Red:
            # within red light triggerbox
            if state.on_red:
                state.on_red_speed.append(speed)
            else:
                state.on_red = True
                state.on_red_speed = list()
    else:
        # not at traffic light
        if state.on_red:
            # out of red light triggerbox
            state.on_red = False
            stopped_at_red = False
            for i, ors in enumerate(state.on_red_speed):
                if ors < 0.1:
                    stopped_at_red = True
            if not stopped_at_red:
                state.red_violation = True
    return (
        frame_speed_lim_changed,
        player_lane_id,
        player_loc,
        player_road_id,
        player_rot,
        speed,
        speed_limit,
        vel,
        player_transform,
    )


def check_destination(
    npc_vehicles,
    npc_now,
    agents_now,
    autoware_stuck,
    conf,
    goal_loc,
    goal_rot,
    player_loc,
    proc_state,
    retval,
    s_started,
    sensors,
    speed,
    state,
    world,
):
    dist_to_goal = player_loc.distance(goal_loc)
    break_flag = False
    wq = []
    # mark goal position
    if (
        conf.agent_type == c.BEHAVIOR
        or conf.agent_type == c.INTERFUSER
        or conf.agent_type == c.LMDRIVE
        or conf.agent_type == c.AUTOWARE_0903
    ):
        delete_indices = []
        for i in range(len(agents_now)):
            if (
                os.environ["ADS"] == "interfuser"
                and isinstance(agents_now[i][0], InterfuserAgent)
                and conf.agent_type == c.INTERFUSER
            ):
                if agents_now[i][0].initialized:
                    wq = agents_now[i][0]._route_planner.route
                else:
                    continue
            elif (
                os.environ["ADS"] == "lmdrive"
                and isinstance(agents_now[i][0], LMDriveAgent)
                and conf.agent_type == c.LMDRIVE
            ):
                if agents_now[i][0].initialized:
                    wq = agents_now[i][0]._route_planner.route
                else:
                    print("LMDrive not initial")
                    continue
            elif conf.agent_type == c.BEHAVIOR:
                wq = agents_now[i][0].get_local_planner()._waypoints_queue
            elif (
                conf.agent_type == c.AUTOWARE_0903
                and agents_now[i][0] == "Autoware_0903"
            ):
                wq = [] if autoware_2409_utils.route_finished else [0]
                # print(f"Autoware'wq is {i} : {wq}")
            if len(wq) == 0:
                if i == 0:
                    if speed < 0.1:
                        if dist_to_goal < 10:
                            print(
                                "\n[*]  Reached the destination dist_to_goal=",
                                dist_to_goal,
                            )
                            retval = 0
                            break_flag = True
                            break
                        else:
                            print(
                                "\n[*] dont Reached the destination dist_to_goal=",
                                dist_to_goal,
                            )
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal
                            retval = 1
                            break_flag = True
                            break
                else:
                    delete_indices.append(i)
        for index in delete_indices:
            delete_npc(agents_now[index][2], npc_vehicles, sensors, agents_now, npc_now)
    return break_flag, retval, autoware_stuck, s_started, wq


def world_reload(npc_list, npc_vehicles, npc_walkers, npc_now, sensors, world):
    try:
        for npc in npc_now:
            npc.instance = None
        for npc in npc_list:
            npc.fresh = True
        for s in sensors:
            s.stop()
            s.destroy()
        for w in npc_walkers:
            w.destroy()
        for v in npc_vehicles:
            v.destroy()

        return True
    except RuntimeError:
        return False


def check_and_remove_excess_images(pattern, max_frames):
    images = sorted(glob.glob(pattern))
    while len(images) > max_frames:
        os.remove(images.pop(0))


def save_behavior_video(carla_error, state):
    print("Saving behavior video")
    # # remove jpg files
    max_frames = c.FRAME_RATE * c.VIDEO_TIME
    if state.crashed and not state.laneinvaded:
        print(f"Saving front camera video for last {c.VIDEO_TIME} second", end=" ")
        check_and_remove_excess_images(
            f"/tmp/fuzzerdata-tmfuzz/front-*.jpg", max_frames
        )
    else:
        print(f"Saving the whole front camera video", end=" ")
    vid_filename = f"/tmp/fuzzerdata-tmfuzz/front.mp4"
    if os.path.exists(vid_filename):
        os.remove(vid_filename)
    cmd_cat = f"cat /tmp/fuzzerdata-tmfuzz/front-*.jpg"
    cmd_ffmpeg = " ".join(
        [
            "ffmpeg",
            "-f image2pipe",
            f"-r {c.FRAME_RATE}",
            "-vcodec mjpeg",
            "-i -",
            "-vcodec libx264",
            "-crf 5",
            vid_filename,
        ]
    )
    cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
    if not carla_error:
        os.system(cmd)
    else:
        print("error:dont save any video")
    cmd = f"rm -f /tmp/fuzzerdata-tmfuzz/front-*.jpg"
    os.system(cmd)
    if state.crashed and not state.laneinvaded:
        print(f"Saving top camera video for last {c.VIDEO_TIME}", end=" ")
        check_and_remove_excess_images(f"/tmp/fuzzerdata-tmfuzz/top-*.jpg", max_frames)
    else:
        print(f"Saving the whole top camera video", end=" ")
    vid_filename = f"/tmp/fuzzerdata-tmfuzz/top.mp4"
    if os.path.exists(vid_filename):
        os.remove(vid_filename)
    cmd_cat = f"cat /tmp/fuzzerdata-tmfuzz/top-*.jpg"
    cmd_ffmpeg = " ".join(
        [
            "ffmpeg",
            "-f image2pipe",
            f"-r {c.FRAME_RATE}",
            "-vcodec mjpeg",
            "-i -",
            "-vcodec libx264",
            "-crf 15",
            vid_filename,
        ]
    )
    cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
    if not carla_error:
        os.system(cmd)
    else:
        print("error:dont save any video")
    cmd = f"rm -f /tmp/fuzzerdata-tmfuzz/top-*.jpg"
    os.system(cmd)


def transform_2_location(transform):
    return carla.Location(
        x=transform.location.x, y=transform.location.y, z=transform.location.z
    )


def ego_initialize(
    agents_now,
    exec_state,
    blueprint_library,
    conf,
    player_bp,
    sensors,
    sp,
    state,
    world,
    wp,
):
    # for autoware
    autoware_container = None
    autoware_state_monitor = None
    town = world.get_map()
    player = None
    if conf.agent_type == c.BEHAVIOR:
        player = world.try_spawn_actor(player_bp, sp)
        ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
        ego.set_instance(player)
        world.tick()  # sync once with simulator
        player.set_simulate_physics(True)
        agent = BehaviorAgent(player, behavior="cautious")
        agent.set_destination(
            start_location=sp.location,
            end_location=wp.location,
        )
        agents_now.append((agent, player, ego))
        print("[+] spawned cautious BehaviorAgent")
    elif conf.agent_type == c.INTERFUSER:
        player = world.try_spawn_actor(player_bp, sp)
        ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
        ego.set_instance(player)
        world.tick()  # sync once with simulator
        player.set_simulate_physics(True)
        agent = InterfuserAgent("InterFuser/team_code/interfuser_config.py")
        trajectory = [transform_2_location(sp), transform_2_location(wp)]
        gps_route, route = interpolate_trajectory_interfuser(world, trajectory)
        agent.set_global_plan(gps_route, route)
        agent_wrapper = AgentWrapper_interfuser(agent)
        agent_wrapper.setup_sensors(player)
        agents_now.append((agent, player, ego))
        print(
            "[+] spawned cautious InterFuser: From %s to %s"
            % (sp.location, wp.location)
        )
    elif conf.agent_type == c.LMDRIVE:
        player = CarlaDataProvider_lmdriver.request_new_actor(
            player_bp.id, sp, rolename="hero"
        )
        ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
        ego.set_instance(player)
        world.tick()  # sync once with simulator
        player.set_simulate_physics(True)
        agent = LMDriveAgent("LMDrive/team_code/lmdriver_config.py")
        agent.town_id = town.replace("Carla/Maps/", "")
        agent.sampled_scenarios = []
        agent.scenario_cofing_name = str(random.randint(1, 2025))
        ego_sp = carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z)
        ego_dp = carla.Location(x=wp.location.x, y=wp.location.y, z=wp.location.z)
        trajectory = [ego_sp, ego_dp]
        gps_route, route = interpolate_trajectory_lmdriver(world, trajectory)
        agent.set_global_plan(gps_route, route)
        agent_wrapper = AgentWrapper_lmdriver(agent)
        agent_wrapper.setup_sensors(player)
        agents_now.append((agent, player, ego))
        print("[+] spawned LMDrive Agent")

    elif conf.agent_type == c.AUTOWARE_0903:
        player_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        player_bp.set_attribute("role_name", "autoware_v1")
        player = world.try_spawn_actor(player_bp, sp)
        try:
            physics_control = player.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            player.apply_physics_control(physics_control)
        except Exception as e:
            print(e)
        ego = NPC(npc_type=c.VEHICLE, spawn_point=sp, npc_id=-1)
        ego.set_instance(player)
        _gnss_sensor = GnssSensor(player, sensor_name="ublox")
        _imu_sensor = IMUSensor(player, sensor_name="tamagawa")
        _lidar_sensor = LidarSensor(player, sensor_name="top")
        _rgb_camera = RgbCamera(player, sensor_name="traffic_light")
        world.tick()
        max_wait = 60
        autoware_2409_utils.start_bridge()  # start bridge in backgroun
        time.sleep(10)
        autoware_2409_utils.start_autoware()  # open autoware
        while True:
            if autoware_2409_utils.autoware_start_ready:
                break
            elif max_wait < 0:
                print("Autoware start fail!")
                raise Exception("Autoware start fail!")
            else:
                time.sleep(1)
                print(f"Waiting for Autoware ready... [{max_wait}s]")
                max_wait -= 1
        autoware_2409_utils.send_goal_pose(wp)
        time.sleep(5)  # wait for goal ready
        autoware_2409_utils.restart_ros_daemon()
        autoware_state_monitor = threading.Thread(
            target=autoware_2409_utils.monitor_routing_state
        )
        autoware_state_monitor.daemon = True
        autoware_state_monitor.start()
        max_wait = 60
        while True:
            if autoware_2409_utils.goal_ready == 1:  # good set
                break
            elif autoware_2409_utils.goal_ready == 2:  # error set
                print("Goal Set is Error")
                raise Exception("Autoware Goad Set Error")
            elif autoware_2409_utils.goal_ready == 0:
                print("Waiting Autoware Set goal...")
                time.sleep(1)
                max_wait -= 1
            if max_wait < 0:
                print("Goooal Set fail")
                raise Exception("Goooal Set fail")
        agents_now.append((None, player, ego))
        print("[+] spawned Autowre 0903 Agent")

    rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_camera_bp.set_attribute("image_size_x", "800")
    rgb_camera_bp.set_attribute("image_size_y", "600")
    rgb_camera_bp.set_attribute("fov", "105")

    camera_tf = carla.Transform(carla.Location(z=1.8))
    camera_front = world.spawn_actor(
        rgb_camera_bp,
        camera_tf,
        attach_to=player,
        attachment_type=carla.AttachmentType.Rigid,
    )

    camera_front.listen(lambda image: _on_front_camera_capture(image, state))
    sensors.append(camera_front)
    camera_tf2 = carla.Transform(carla.Location(z=50.0), carla.Rotation(pitch=-90.0))
    camera_top = world.spawn_actor(
        rgb_camera_bp,
        camera_tf2,
        attach_to=player,
        attachment_type=carla.AttachmentType.Rigid,
    )

    camera_top.listen(lambda image: _on_top_camera_capture(image, state))
    sensors.append(camera_top)

    ego.attach_collision(world, sensors, state)
    if conf.check_dict["lane"]:
        ego.attach_lane_invasion(world, sensors, state)
    # get vehicle's maximum steering angle
    physics_control = player.get_physics_control()
    max_steer_angle = 0
    for wheel in physics_control.wheels:
        if wheel.max_steer_angle > max_steer_angle:
            max_steer_angle = wheel.max_steer_angle
    if conf.agent_type != c.AUTOWARE_0903:
        world.tick()  # sync with simulator
        print("Tick!")
    # while True:
    #         print("!!")
    #         time.sleep(1)
    return autoware_container, ego, player, max_steer_angle, autoware_state_monitor


def simulate_initialize(client, conf, weather_dict, world):
    # client.set_timeout(10.0)
    if conf.no_traffic_lights:
        set_traffic_lights_state(world, carla.TrafficLightState.Green)
        world.freeze_all_traffic_lights(True)

    # if conf.debug:
    #     print("[debug] world:", world)
    # else:
    #     world.reset_all_traffic_lights()
    town_map = world.get_map()
    if conf.debug:
        print("[debug] map:", town_map)
    blueprint_library = world.get_blueprint_library()
    vehicle_bp_library = blueprint_library.filter("vehicle.*")
    walker_bp = blueprint_library.find("walker.pedestrian.0001")  # 0001~0014
    walker_controller_bp = blueprint_library.find("controller.ai.walker")
    player_bp = blueprint_library.filter("nissan")[0]
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE  # FPS
    # settings.no_rendering_mode = False
    # world.apply_settings(settings)
    frame_id = world.tick()
    clock = pygame.time.Clock()
    if conf.density != 0:
        add_car_frame = c.FRAME_RATE // conf.density
    else:
        add_car_frame = 9999999
    # set weather
    weather = world.get_weather()
    weather.cloudiness = weather_dict["cloud"]
    weather.precipitation = weather_dict["rain"]
    # weather.precipitation_deposits = weather_dict["puddle"]
    weather.wetness = weather_dict["wetness"]
    weather.wind_intensity = weather_dict["wind"]
    weather.fog_density = weather_dict["fog"]
    weather.sun_azimuth_angle = weather_dict["angle"]
    weather.sun_altitude_angle = weather_dict["altitude"]
    world.set_weather(weather)
    world.tick()  # sync with simulator
    return (
        add_car_frame,
        blueprint_library,
        clock,
        player_bp,
        town_map,
        vehicle_bp_library,
    )


def spawn_npc(
    npc,
    npc_vehicle,
    npc_vehicles,
    npcs_now,
    agents_now,
    conf,
    max_wheels_for_non_motorized,
    road_direction,
    sensors,
    state,
    world,
    wp,
):
    npc_vehicles.append(npc_vehicle)
    npc_vehicle.set_transform(npc.spawn_point)
    x_offset = random.uniform(5, 10)
    y_offset = random.uniform(5, 10)
    wp_new_location = wp.location + carla.Location(
        x=x_offset * random.choice([-1, 1]), y=y_offset * random.choice([-1, 1])
    )
    new_agent = set_autopilot(
        npc_vehicle, c.BEHAVIOR_AGENT, npc.spawn_point.location, wp_new_location, world
    )
    agents_now.append((new_agent, npc_vehicle, npc))
    npc_vehicle.set_target_velocity(npc.speed * road_direction)
    npc.set_instance(npc_vehicle)
    # # just add it for behavior
    # if conf.agent_type == c.BEHAVIOR:
    #     # don't add sensors for non_motorized vehicles
    #     if npc.npc_bp.get_attribute(
    #             "number_of_wheels").as_int() > max_wheels_for_non_motorized:
    #         npc.attach_collision(world, sensors, state)
    npcs_now.append(npc)
    npc.fresh = False


def _on_front_camera_capture(image, state):
    if not state.end:
        image.save_to_disk(f"/tmp/fuzzerdata-tmfuzz/front-{image.frame:05d}.jpg")


def _on_top_camera_capture(image, state):
    if not state.end:
        image.save_to_disk(f"/tmp/fuzzerdata-tmfuzz/top-{image.frame:05d}.jpg")


def _set_camera(conf, player, spectator):
    if conf.view == c.BIRDSEYE:
        _cam_over_player(player, spectator)
    elif conf.view == c.ONROOF:
        _cam_chase_player(player, spectator)
    else:  # fallthrough default
        _cam_chase_player(player, spectator)


def _cam_chase_player(player, spectator):
    location = player.get_location()
    rotation = player.get_transform().rotation
    fwd_vec = rotation.get_forward_vector()

    # chase from behind
    constant = 4
    location.x -= constant * fwd_vec.x
    location.y -= constant * fwd_vec.y
    # and above
    location.z += 3
    rotation.pitch -= 5
    spectator.set_transform(carla.Transform(location, rotation))


def _cam_over_player(player, spectator):
    location = player.get_location()
    location.z += 100
    # rotation = player.get_transform().rotation
    rotation = carla.Rotation()  # fix rotation for better sim performance
    rotation.pitch -= 90
    spectator.set_transform(carla.Transform(location, rotation))


def non_blocking_read(stdout):
    fd = stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    output_state = b""
    rlist, _, _ = select.select([stdout], [], [], 0.01)
    if stdout in rlist:
        data = os.read(fd, 4096)
        output_state = data
    output_state = output_state.decode("utf-8")
    return output_state


def check_violation(
    conf,
    cur_frame_id,
    frame_speed_lim_changed,
    retval,
    speed,
    speed_limit,
    state,
    wait_until_end,
    snapshot,
    player_transform,
):
    # Check speeding
    if conf.check_dict["speed"]:
        # allow T seconds to slow down if speed limit suddenly
        # decreases
        T = 3  # 0 for strict checking
        if (
            speed > speed_limit + 2
            and cur_frame_id > frame_speed_lim_changed + T * c.FRAME_RATE
        ):
            print(
                "\n[*] Speed violation: {} km/h on a {} km/h road".format(
                    speed, speed_limit
                )
            )
            state.speeding = True
            state.speeding_details.append((snapshot.timestamp, player_transform))
            retval = 1
            wait_until_end = 1
    # Check crash
    if conf.check_dict["crash"]:
        if state.crashed:
            print("\n[*] Collision detected: %.2f" % (state.elapsed_time))
            retval = 1
            wait_until_end = 1
    # Check lane violation
    if conf.check_dict["lane"]:
        if state.laneinvaded:
            retval = 1
    # Check traffic light violation
    if conf.check_dict["red"]:
        if state.red_violation:
            print("\n[*] Red light violation detected: %.2f" % (state.elapsed_time))
            retval = 1
            state.running_red_light_details.append(
                (snapshot.timestamp, player_transform)
            )
            wait_until_end = 1
    # Check inactivity
    if speed < 1:  # km/h
        state.stuck_duration += 1
    else:
        state.stuck_duration = 0
    if conf.check_dict["stuck"]:
        if state.stuck_duration > (60 * c.FRAME_RATE):
            state.stuck = True
            print("\n[*] Stuck for too long: %d" % state.stuck_duration)
            state.stuck_details.append((snapshot.timestamp, player_transform))
            retval = 1
            wait_until_end = 1
    if conf.check_dict["other"]:
        if state.num_frames > 60 * c.FRAME_RATE * 15:  # over 15 minutes
            print("\n[*] Simulation taking too long")
            state.other_error = "timeout"
            state.other_error_val = state.num_frames
            retval = 1
            wait_until_end = 1
        if state.other_error:
            print("\n[*] Other error: %d" % state.signal)
            retval = 1
            wait_until_end = 1
    return retval, wait_until_end


def run_cmd_in_container(container, cmd):
    container.exec_run(cmd, stdout=True, stderr=True, user="root")


def check_topo(player_waypoint=None, waypoint=None, G=None):
    player_lane_id = player_waypoint.lane_id
    player_road_id = player_waypoint.road_id
    neighbors_A = nx.single_source_shortest_path_length(
        G, source=(player_road_id, player_lane_id), cutoff=2
    )
    neighbors_A[(player_road_id, player_lane_id)] = 0
    neighbors_B = nx.single_source_shortest_path_length(
        G, source=(waypoint.road_id, waypoint.lane_id), cutoff=2
    )
    neighbors_B[(waypoint.road_id, waypoint.lane_id)] = 0
    if not any(node in neighbors_A and node in neighbors_B for node in G.nodes()):
        return False
    else:
        return True


def check_vehicle(proc):
    while True:
        output_state = non_blocking_read(proc.stdout)
        if "VehicleReady" in output_state:
            break
        time.sleep(1)
        print("[*] Waiting for Autoware vehicle Ready" + "\r", end="")


def location_pickle(location):
    return carla.Location, (location.x, location.y, location.z)


def rotation_pickle(rotation):
    return carla.Rotation, (rotation.pitch, rotation.yaw, rotation.roll)


def transform_pickle(transform):
    return carla.Transform, (
        carla.Location(
            transform.location.x, transform.location.y, transform.location.z
        ),
        carla.Rotation(
            transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll
        ),
    )


def actorblueprint_pickle(actorblueprint):
    return carla.ActorBlueprint, (actorblueprint.id, actorblueprint.tags)


def timestamp_pickle(timestamp):
    return carla.Timestamp, (
        timestamp.frame,
        timestamp.elapsed_seconds,
        timestamp.delta_seconds,
        timestamp.platform_timestamp,
    )


class CustomPickler(dill.Pickler):
    def save(self, obj):
        if isinstance(obj, carla.libcarla.Vehicle):
            # 将Vehicle实例替换为None
            self.save_reduce(lambda: None, ())
        else:
            # 其他对象正常处理
            super().save(obj)


# 保存变量时使用自定义Pickler
def save_variable(var, filename):
    with open(filename, "wb") as f:
        CustomPickler(f).dump(var)

if __name__ == "__main__":
    print("Start simulate.py")
    copyreg.pickle(carla.Location, location_pickle)
    copyreg.pickle(carla.Rotation, rotation_pickle)
    copyreg.pickle(carla.Transform, transform_pickle)
    copyreg.pickle(carla.ActorBlueprint, actorblueprint_pickle)
    copyreg.pickle(carla.Timestamp, timestamp_pickle)
    param = dill.load(open("param.pick", "rb"))
    ret, npc_list, state = simulate(*param)
    dill.dump(ret, open("ret.pick", "wb"))
    for n in npc_list:
        if n.npc_bp != None:
            n.npc_bp = n.npc_bp.id
    save_variable(npc_list, "npc.pick")
    state.closest_car = None
    dill.dump(state, open("state.pick", "wb"))
    print("Finished simulate.py")
