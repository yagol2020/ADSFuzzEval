import random

import coverage

cov = coverage.Coverage()
cov.start()
########################
import copyreg
import os
import shutil
import subprocess
import sys
import signal
import time
import math
import traceback

import constants as c

import carla

from oracle import AllOracle
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
import pygame
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


def get_carla_transform(loc_rot_tuples):
    """
    Convert loc_rot_tuples = ((x, y, z), (roll, pitch, yaw)) to
    carla.Transform object
    """

    if loc_rot_tuples is None:
        return None

    loc = loc_rot_tuples[0]
    rot = loc_rot_tuples[1]

    t = carla.Transform(
        carla.Location(loc[0], loc[1], loc[2]),
        carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
    )

    return t


def _on_collision(event, state):
    if state.simulation_running is False:
        return

    if event.other_actor.type_id != "static.road":
        state.crashed = True
        state.collision_details.append((event.timestamp, event.transform))


def _on_invasion(event, state):
    if state.simulation_running is False:
        return
    crossed_lanes = event.crossed_lane_markings
    for crossed_lane in crossed_lanes:
        if crossed_lane.lane_change == carla.LaneChange.NONE:
            state.laneinvaded = True
            state.laneinvasion_details.append((event.timestamp, event.transform))


def _on_front_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata-drivefuzz/front-{image.frame}.jpg")


def _on_top_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata-drivefuzz/top-{image.frame}.jpg")


def set_camera(conf, player, spectator):
    if conf.view == c.BIRDSEYE:
        cam_over_player(player, spectator)
    elif conf.view == c.ONROOF:
        cam_chase_player(player, spectator)
    else:  # fallthru default
        cam_chase_player(player, spectator)


def cam_chase_player(player, spectator):
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


def cam_over_player(player, spectator):
    location = player.get_location()
    location.z += 100
    # rotation = player.get_transform().rotation
    rotation = carla.Rotation()  # fix rotation for better sim performance
    rotation.pitch -= 90
    spectator.set_transform(carla.Transform(location, rotation))


def is_player_on_puddle(player_loc, actor_frictions):
    for friction in actor_frictions:
        len_x = float(friction.attributes["extent_x"])
        len_y = float(friction.attributes["extent_y"])
        loc_x = friction.get_location().x
        loc_y = friction.get_location().y
        p1 = loc_x - len_x / 100
        p2 = loc_x + len_x / 100
        p3 = loc_y - len_y / 100
        p4 = loc_y + len_y / 100
        p_x = player_loc.x
        p_y = player_loc.y
        if p1 <= p_x and p_x <= p2 and p3 <= p_y and p_y <= p4:
            return True
        else:
            return False


def transform_2_location(transform):
    return carla.Location(
        x=transform.location.x, y=transform.location.y, z=transform.location.z
    )


def simulate(conf, state, town, sp, wp, weather_dict, frictions_list, actors_list):
    carla_port_docker = os.environ.get("CARLA_PORT_DOCKER", 2000)
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
        client.set_timeout(10.0)
        world = client.load_world(str(town))
        tm = client.get_trafficmanager(int(carla_port_docker) + 2025)
    except Exception as carla_connect_error:
        print(carla_connect_error)
        subprocess.Popen(
            ["./run_carla_v2.sh", "-p", str(carla_port_docker), "-v", carla_v, "-d"]
        )
        time.sleep(10)
        return simulate(
            conf, state, town, sp, wp, weather_dict, frictions_list, actors_list
        )
    retval = 0
    if os.path.exists("/tmp/fuzzerdata-drivefuzz"):
        shutil.rmtree("/tmp/fuzzerdata-drivefuzz")
    os.mkdir("/tmp/fuzzerdata-drivefuzz")
    state.location_recorder = []
    #####################
    state.running_red_light_details = []
    state.stuck_details = []
    state.speeding_details = []
    state.collision_details = []
    state.laneinvasion_details = []
    state.start_simulate_time = time.time()
    state.start_simulate_time_world = world.get_snapshot().timestamp.elapsed_seconds
    state.simulation_running = True
    #####################
    agent = None
    agent_wrapper = None
    all_oracle = None
    #####################
    try:
        blueprint_library = world.get_blueprint_library()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE  # FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        frame_id = world.tick()
        init_frame_id = frame_id
        frame_0 = frame_id
        start_time = time.time()
        clock = pygame.time.Clock()

        # set weather
        weather = world.get_weather()
        weather.cloudiness = weather_dict["cloud"]
        weather.precipitation = weather_dict["rain"]
        weather.precipitation_deposits = weather_dict["puddle"]
        weather.wetness = weather_dict["wetness"]
        weather.wind_intensity = weather_dict["wind"]
        weather.fog_density = weather_dict["fog"]
        weather.sun_azimuth_angle = weather_dict["angle"]
        weather.sun_altitude_angle = weather_dict["altitude"]
        world.set_weather(weather)
        if conf.agent_type == c.INTERFUSER:
            CarlaDataProvider_interfuser.set_client(client)
            CarlaDataProvider_interfuser.set_world(world)
        elif conf.agent_type == c.LMDRIVE:
            CarlaDataProvider_lmdriver.cleanup()
            CarlaDataProvider_lmdriver.set_client(client)
            CarlaDataProvider_lmdriver.set_world(world)
            GameTime_lmdriver.restart()
        elif conf.agent_type == c.AUTOWARE_0903:
            autoware_2409_utils.stop_autoware()
            autoware_2409_utils.stop_bridge()
            # autoware_2409_utils.zero_counters()

        sensors = []
        actor_vehicles = []
        actor_walkers = []
        actor_controllers = []
        actor_frictions = []
        ros_pid = 0

        world.tick()  # sync once with simulator

        # spawn player
        # how DriveFuzz spawns a player vehicle depends on
        # the autonomous driving agent
        player_bp = blueprint_library.filter("mercedes")[0]
        # player_bp.set_attribute("role_name", "ego")
        player = None

        goal_loc = wp.location
        goal_rot = wp.rotation

        # mark goal position
        if conf.debug:
            world.debug.draw_box(
                box=carla.BoundingBox(goal_loc, carla.Vector3D(0.2, 0.2, 1.0)),
                rotation=goal_rot,
                life_time=0,
                thickness=1.0,
                color=carla.Color(r=0, g=255, b=0),
            )

        if conf.agent_type == c.BASIC:
            player = world.try_spawn_actor(player_bp, sp)
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0  # player
                retval = -1
                return  # trap to finally

            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)

            agent = BasicAgent(player)
            agent.set_destination((wp.location.x, wp.location.y, wp.location.z))
            print("[+] spawned BasicAgent")

        elif conf.agent_type == c.BEHAVIOR:
            player = world.try_spawn_actor(player_bp, sp)
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0  # player
                retval = -1
                return  # trap to finally

            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)

            agent = BehaviorAgent(player, behavior="cautious")
            agent.set_destination(
                start_location=sp.location,
                end_location=wp.location,
            )

            print("[+] spawned cautious BehaviorAgent")
        elif conf.agent_type == c.INTERFUSER:
            player = world.try_spawn_actor(player_bp, sp)
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0  # player
                retval = -1
                return  # trap to finally

            world.tick()  # sync once with simulator
            player.set_simulate_physics(True)

            agent = InterfuserAgent("InterFuser/team_code/interfuser_config.py")
            trajectory = [transform_2_location(sp), transform_2_location(wp)]
            gps_route, route = interpolate_trajectory_interfuser(world, trajectory)
            agent.set_global_plan(gps_route, route)
            agent_wrapper = AgentWrapper_interfuser(agent)
            agent_wrapper.setup_sensors(player)
            print("[+] spawned cautious InterFuser")
        elif conf.agent_type == c.LMDRIVE:
            player = CarlaDataProvider_lmdriver.request_new_actor(
                player_bp.id, sp, rolename="hero"
            )
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0  # player
                retval = -1
                return  # trap to finally

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
        # Attach collision detector
        collision_bp = blueprint_library.find("sensor.other.collision")
        sensor_collision = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=player
        )
        sensor_collision.listen(lambda event: _on_collision(event, state))
        sensors.append(sensor_collision)

        # Attach lane invasion sensor
        lanesensor_bp = blueprint_library.find("sensor.other.lane_invasion")
        sensor_lane = world.spawn_actor(
            lanesensor_bp, carla.Transform(), attach_to=player
        )
        sensor_lane.listen(lambda event: _on_invasion(event, state))
        sensors.append(sensor_lane)

        if (
            conf.agent_type == c.BASIC
            or conf.agent_type == c.BEHAVIOR
            or conf.agent_type == c.INTERFUSER
            or conf.agent_type == c.LMDRIVE
            or conf.agent_type == c.AUTOWARE_0903
        ):
            # Attach RGB camera (front)
            rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")

            rgb_camera_bp.set_attribute("image_size_x", "800")
            rgb_camera_bp.set_attribute("image_size_y", "600")
            rgb_camera_bp.set_attribute("fov", "105")

            # position relative to the parent actor (player)
            camera_tf = carla.Transform(carla.Location(z=1.8))

            # time in seconds between sensor captures - should sync w/ fps?
            # rgb_camera_bp.set_attribute("sensor_tick", "1.0")

            camera_front = world.spawn_actor(
                rgb_camera_bp,
                camera_tf,
                attach_to=player,
                attachment_type=carla.AttachmentType.Rigid,
            )

            camera_front.listen(lambda image: _on_front_camera_capture(image))

            sensors.append(camera_front)

            camera_tf = carla.Transform(
                carla.Location(z=50.0), carla.Rotation(pitch=-90.0)
            )
            camera_top = world.spawn_actor(
                rgb_camera_bp,
                camera_tf,
                attach_to=player,
                attachment_type=carla.AttachmentType.Rigid,
            )

            camera_top.listen(lambda image: _on_top_camera_capture(image))
            sensors.append(camera_top)

        world.tick()  # sync with simulator

        # get vehicle's maximum steering angle
        physics_control = player.get_physics_control()
        max_steer_angle = 0
        for wheel in physics_control.wheels:
            if wheel.max_steer_angle > max_steer_angle:
                max_steer_angle = wheel.max_steer_angle

        # spawn friction triggers
        friction_bp = blueprint_library.find("static.trigger.friction")
        for friction in frictions_list:
            friction_bp.set_attribute("friction", str(friction["level"]))
            friction_bp.set_attribute("extent_x", str(friction["size"][0]))
            friction_bp.set_attribute("extent_y", str(friction["size"][1]))
            friction_bp.set_attribute("extent_z", str(friction["size"][2]))

            friction_sp_transform = get_carla_transform(friction["spawn_point"])
            friction_size_loc = carla.Location(
                friction["size"][0], friction["size"][1], friction["size"][2]
            )

            friction_trigger = world.try_spawn_actor(friction_bp, friction_sp_transform)

            if friction_trigger is None:
                print(
                    "[-] Failed spawning lvl {} puddle at ({}, {})".format(
                        friction["level"],
                        friction_sp_transform.location.x,
                        friction_sp_transform.location.y,
                    )
                )

                state.spawn_failed = True
                state.spawn_failed_object = friction
                retval = -1
                return
            actor_frictions.append(friction_trigger)  # to destroy later

            # Optional for visualizing trigger (for debugging)
            if conf.debug:
                world.debug.draw_box(
                    box=carla.BoundingBox(
                        friction_sp_transform.location, friction_size_loc * 1e-2
                    ),
                    rotation=friction_sp_transform.rotation,
                    life_time=0,
                    thickness=friction["level"] * 1,  # the stronger the thicker
                    color=carla.Color(r=0, g=0, b=255),
                )
            print(
                "[+] New puddle [%d] @(%.2f, %.2f) lvl %.2f"
                % (
                    friction_trigger.id,
                    friction_sp_transform.location.x,
                    friction_sp_transform.location.y,
                    friction["level"],
                )
            )

        # spawn actors
        vehicle_bp = blueprint_library.find("vehicle.bmw.grandtourer")
        vehicle_bp.set_attribute("color", "255,0,0")
        walker_bp = blueprint_library.find("walker.pedestrian.0001")  # 0001~0014
        walker_controller_bp = blueprint_library.find("controller.ai.walker")

        for actor in actors_list:
            actor_sp = get_carla_transform(actor["spawn_point"])
            if actor["type"] == c.VEHICLE:  # vehicle
                actor_vehicle = world.try_spawn_actor(vehicle_bp, actor_sp)
                actor_nav = c.NAVTYPE_NAMES[actor["nav_type"]]
                if actor_vehicle is None:
                    actor_str = c.ACTOR_NAMES[actor["type"]]
                    print(
                        "[-] Failed spawning {} {} at ({}, {})".format(
                            actor_nav,
                            actor_str,
                            actor_sp.location.x,
                            actor_sp.location.y,
                        )
                    )

                    state.spawn_failed = True
                    state.spawn_failed_object = actor
                    retval = -1
                    return  # trap to finally

                actor_vehicles.append(actor_vehicle)
                print(
                    "[+] New %s vehicle [%d] @(%.2f, %.2f) yaw %.2f"
                    % (
                        actor_nav,
                        actor_vehicle.id,
                        actor_sp.location.x,
                        actor_sp.location.y,
                        actor_sp.rotation.yaw,
                    )
                )

            elif actor["type"] == c.WALKER:  # walker
                actor_walker = world.try_spawn_actor(walker_bp, actor_sp)
                actor_nav = c.NAVTYPE_NAMES[actor["nav_type"]]
                if actor_walker is None:
                    actor_str = c.ACTOR_NAMES[actor["type"]]
                    print(
                        "[-] Failed spawning {} {} at ({}, {})".format(
                            actor_nav,
                            actor_str,
                            actor_sp.location.x,
                            actor_sp.location.y,
                        )
                    )

                    state.spawn_failed = True
                    state.spawn_failed_object = actor
                    retval = -1
                    return  # trap to finally

                actor_walkers.append(actor_walker)
                print(
                    "[+] New %s walker [%d] @(%.2f, %.2f) yaw %.2f"
                    % (
                        actor_nav,
                        actor_walker.id,
                        actor_sp.location.x,
                        actor_sp.location.y,
                        actor_sp.rotation.yaw,
                    )
                )
        # print("after spawning actors", time.time())

        # print("real simulation begins", time.time())
        cnt_v = 0
        cnt_w = 0
        for actor in actors_list:
            actor_sp = get_carla_transform(actor["spawn_point"])
            actor_dp = get_carla_transform(actor["dest_point"])
            if actor["type"] == c.VEHICLE:  # vehicle
                actor_vehicle = actor_vehicles[cnt_v]
                cnt_v += 1
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * actor["speed"])

                elif actor["nav_type"] == c.MANEUVER:
                    # set initial speed 0
                    # trajectory control happens in the control loop below
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * 0)

                elif actor["nav_type"] == c.AUTOPILOT:
                    actor_vehicle.set_autopilot(True, tm.get_port())

                actor_vehicle.set_simulate_physics(True)
            elif actor["type"] == c.WALKER:  # walker
                actor_walker = actor_walkers[cnt_w]
                cnt_w += 1
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    controller_walker = carla.WalkerControl()
                    controller_walker.direction = forward_vec
                    controller_walker.speed = actor["speed"]
                    actor_walker.apply_control(controller_walker)

                elif actor["nav_type"] == c.AUTOPILOT:
                    controller_walker = world.spawn_actor(
                        walker_controller_bp, actor_sp, actor_walker
                    )

                    world.tick()  # without this, walker vanishes
                    controller_walker.start()
                    controller_walker.set_max_speed(float(actor["speed"]))
                    controller_walker.go_to_location(actor_dp.location)

                elif actor["nav_type"] == c.IMMOBILE:
                    controller_walker = None

                if controller_walker:  # can be None if immobile walker
                    actor_controllers.append(controller_walker)

        elapsed_time = 0
        start_time = time.time()

        yaw = sp.rotation.yaw

        player_loc = player.get_transform().location
        init_x = player_loc.x
        init_y = player_loc.y

        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGSEGV, state.sig_handler)
        signal.signal(signal.SIGABRT, state.sig_handler)

        try:
            # actual monitoring of the driving simulation begins here
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
            s_started = False
            s_stopped_frames = 0
            if conf.agent_type == c.AUTOWARE_0903:
                if (
                    not autoware_2409_utils.change_to_autonomous_mode()
                ):  # Set to Autopilot
                    print("Set autopilot Error! Exit")
                    raise Exception("Autoware Set Autopilot Error")
            if conf.debug:
                print("[*] START DRIVING: {} {}".format(first_frame_id, first_sim_time))
            print("executor.py start loop")
            start_loop_time = time.time()
            all_oracle = AllOracle(player, world, conf.out_dir)
            while time.time() - start_loop_time < 10 * 60:
                # Use sampling frequency of FPS*2 for precision!
                clock.tick(c.FRAME_RATE * 2)

                # Carla agents are running in synchronous mode,
                # so we need to send ticks. Not needed for Autoware
                if (
                    conf.agent_type == c.BASIC
                    or conf.agent_type == c.BEHAVIOR
                    or conf.agent_type == c.INTERFUSER
                    or conf.agent_type == c.LMDRIVE
                ):
                    world.tick()

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

                player_transform = player.get_transform()
                player_loc = player_transform.location
                player_rot = player_transform.rotation

                # Get speed
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

                print(
                    "(%.2f,%.2f)>(%.2f,%.2f)>(%.2f,%.2f) %.2f m left, %.2f/%d km/h"
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
                    )
                )

                state.location_recorder.append((player_loc.x, player_loc.y))
                ############################
                carla_spectator = world.get_spectator()
                camera_location = player_loc + carla.Location(z=20)
                camera_rotation = carla.Rotation(pitch=-90)
                carla_spectator.set_transform(
                    carla.Transform(camera_location, camera_rotation)
                )
                ############################

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

                if conf.agent_type == c.BASIC:
                    control = agent.run_step(debug=True)
                    player.apply_control(control)

                elif conf.agent_type == c.BEHAVIOR:
                    agent._update_information()
                    agent.get_local_planner().set_speed(speed_limit)

                    control = agent.run_step()
                    player.apply_control(control)

                elif conf.agent_type == c.INTERFUSER:
                    control = agent()
                    player.apply_control(control)
                elif conf.agent_type == c.LMDRIVE:
                    control = agent()
                    player.apply_control(control)
                elif conf.agent_type == c.AUTOWARE_0903:  # Autoware's control
                    control = player.get_control()
                state.cont_throttle.append(control.throttle)
                state.cont_brake.append(control.brake)
                state.cont_steer.append(control.steer)
                steer_angle = control.steer * max_steer_angle
                state.steer_angle_list.append(steer_angle)

                current_yaw = player_rot.yaw
                state.yaw_list.append(current_yaw)

                yaw_diff = current_yaw - yaw
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                elif yaw_diff < -180:
                    yaw_diff = 360 + yaw_diff

                yaw_rate = yaw_diff * c.FRAME_RATE
                state.yaw_rate_list.append(yaw_rate)
                yaw = current_yaw

                # uncomment below to follow along the player
                # set_camera(conf, player, spectator)

                for v in actor_vehicles:
                    dist = player_loc.distance(v.get_location())
                    if dist < state.min_dist:
                        state.min_dist = dist

                for w in actor_walkers:
                    dist = player_loc.distance(w.get_location())
                    if dist < state.min_dist:
                        state.min_dist = dist

                # Get the lateral speed
                player_right_vec = player_rot.get_right_vector()
                lat_speed = abs(vel.x * player_right_vec.x + vel.y * player_right_vec.y)
                lat_speed *= 3.6  # m/s to km/h
                state.lat_speed_list.append(lat_speed)

                player_fwd_vec = player_rot.get_forward_vector()
                lon_speed = abs(vel.x * player_fwd_vec.x + vel.y * player_fwd_vec.y)
                lon_speed *= 3.6
                state.lon_speed_list.append(lon_speed)

                # Handle actor maneuvers
                if conf.strategy == c.TRAJECTORY:
                    actor = actors_list[0]
                    maneuvers = actor["maneuvers"]

                    maneuver_id = int(state.num_frames / c.FRAMES_PER_TIMESTEP)
                    if maneuver_id < 5:
                        maneuver = maneuvers[maneuver_id]

                        if maneuver[2] == 0:
                            maneuver[2] = state.num_frames
                            actor_vehicle = actor_vehicles[0]

                            # perform the action
                            actor_direction = maneuver[0]
                            actor_speed = maneuver[1]

                            forward_vec = get_carla_transform(
                                actor["spawn_point"]
                            ).rotation.get_forward_vector()

                            if actor_direction == 0:  # forward

                                actor_vehicle.set_target_velocity(
                                    forward_vec * actor_speed
                                )

                        elif (
                            maneuver[2] > 0 and abs(maneuver[2] - state.num_frames) < 40
                        ):
                            # continuously apply lateral force to the vehicle
                            # for 40 frames (2 secs)
                            actor_direction = maneuver[0]
                            apex_degree = maneuver[1]

                            """
                            Model smooth lane changing through varying thetas
                            (theta)
                            45           * *
                            30       * *     * *
                            15     *             * *
                            0  * *                   *
                               0 5 10 15 20 25 30 35 40 (t = # frame)
                            """

                            theta_max = apex_degree
                            force_constant = 5  # should weigh by actor_speed?

                            t = abs(maneuver[2] - state.num_frames)
                            if t < 20:
                                theta = t * (theta_max / 20)
                            else:
                                theta = t * -1 * (theta_max / 20) + 2 * theta_max

                            if actor_direction != 0:  # skip if fwd
                                if actor_direction == -1:  # switch to left lane
                                    theta *= -1  # turn cc-wise
                                elif actor_direction == 1:  # switch to right lane
                                    pass  # turn c-wise

                                theta_rad = math.radians(theta)
                                sin = math.sin(theta_rad)
                                cos = math.cos(theta_rad)

                                x0 = forward_vec.x
                                y0 = forward_vec.y

                                x1 = cos * x0 - sin * y0
                                y1 = sin * x0 + cos * y0

                                dir_vec = carla.Vector3D(x=x1, y=y1, z=0.0)
                                actor_vehicle.set_target_velocity(
                                    dir_vec * force_constant
                                )

                # Check destination
                dist_to_goal = player_loc.distance(goal_loc)
                waypoint_queue = []
                if conf.agent_type == c.BASIC:
                    if hasattr(agent, "done") and agent.done():
                        print("\n[*] (BasicAgent) Reached the destination")
                        if dist_to_goal > 2 and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal
                        break

                elif conf.agent_type == c.BEHAVIOR:
                    lp = agent.get_local_planner()
                    if len(lp._waypoints_queue) == 0:
                        print("\n[*] (BehaviorAgent) Reached the destination")

                        if dist_to_goal > 2 and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal
                        break
                elif conf.agent_type == c.INTERFUSER:
                    waypoint_queue = agent._route_planner.route
                    if len(agent._route_planner.route) == 0:
                        print("\n[*] (InterFuser) Reached the destination")

                        if dist_to_goal > 2 and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal
                        break
                elif conf.agent_type == c.LMDRIVE:
                    if agent.initialized:
                        waypoint_queue = agent._route_planner.route
                        if len(waypoint_queue) == 0:
                            print("\n[*] (LMDrive) Reached the destination")
                            if dist_to_goal > 2 and state.num_frames > 300:
                                state.other_error = "goal"
                                state.other_error_val = dist_to_goal
                            break
                elif conf.agent_type == c.AUTOWARE_0903:
                    waypoint_queue = [] if autoware_2409_utils.route_finished else [0]
                    if autoware_2409_utils.route_finished:
                        print("Autoware finished route!")
                        break
                if dist_to_goal < 5:
                    print("\n[*] (Carla heuristic) Reached the destination")
                    retval = 0
                    break
                all_oracle.update(snapshot, goal_loc, waypoint_queue)
                # Check speeding
                if conf.check_dict["speed"]:
                    T = 3  # 0 for strict checking
                    if (
                        speed > speed_limit
                        and cur_frame_id > frame_speed_lim_changed + T * c.FRAME_RATE
                    ):
                        print(
                            "\n[*] Speed violation: {} km/h on a {} km/h road".format(
                                speed, speed_limit
                            )
                        )
                        state.speeding = True
                        state.speeding_details.append(
                            (snapshot.timestamp, player_transform)
                        )
                        retval = 1
                        break

                # Check crash
                if conf.check_dict["crash"]:
                    if state.crashed:
                        print("\n[*] Collision detected: %.2f" % (state.elapsed_time))
                        retval = 1
                        break

                # Check lane violation
                if conf.check_dict["lane"]:
                    if state.laneinvaded:
                        print(
                            "\n[*] Lane invasion detected: %.2f" % (state.elapsed_time)
                        )
                        retval = 1
                        break

                # Check traffic light violation
                if conf.check_dict["red"]:
                    if state.red_violation:
                        print(
                            "\n[*] Red light violation detected: %.2f"
                            % (state.elapsed_time)
                        )
                        state.running_red_light_details.append(
                            (snapshot.timestamp, player_transform)
                        )
                        retval = 1
                        break

                # Check inactivity
                if speed < 1:  # km/h
                    state.stuck_duration += 1
                else:
                    state.stuck_duration = 0

                if conf.check_dict["stuck"]:
                    if state.stuck_duration > (
                        60 * c.FRAME_RATE
                    ):  # timout default is 60
                        state.stuck = True
                        print("\n[*] Stuck for too long: %d" % (state.stuck_duration))
                        state.stuck_details.append(
                            (snapshot.timestamp, player_transform)
                        )
                        retval = 1
                        break

                if conf.check_dict["other"]:
                    if state.num_frames > 12000:  # over 10 minutes
                        print("\n[*] Simulation taking too long")
                        state.other_error = "timeout"
                        state.other_error_val = state.num_frames
                        retval = 1
                        break
                    if state.other_error:
                        print("\n[*] Other error: %d" % (state.signal))
                        retval = 1
                        break

        except KeyboardInterrupt:
            print("quitting")
            retval = 128
        return

    except Exception as e:
        print("[-] Runtime error:")
        traceback.print_exc()
        retval = 1

    finally:
        state.simulation_running = False
        state.end_simulate_time = time.time()
        state.end_simulate_time_world = world.get_snapshot().timestamp.elapsed_seconds
        print(
            "[*] End simulation: rate %.2f"
            % (
                (state.end_simulate_time - state.start_simulate_time)
                / (state.end_simulate_time_world - state.start_simulate_time_world)
            )
        )
        if conf.agent_type == c.AUTOWARE_0903:
            autoware_2409_utils.monitor_routing_event.set()
            autoware_2409_utils.stop_autoware()
            autoware_2409_utils.stop_bridge()
        if all_oracle:
            all_oracle.save_2_file()
            if cov:
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
        if (
            conf.agent_type == c.BASIC
            or conf.agent_type == c.BEHAVIOR
            or conf.agent_type == c.INTERFUSER
            or conf.agent_type == c.LMDRIVE
            or conf.agent_type == c.AUTOWARE_0903
        ):
            print("Saving front camera video", end=" ")
            vid_filename = "/tmp/fuzzerdata-drivefuzz/front.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)
            cmd_cat = "ls /tmp/fuzzerdata-drivefuzz/front-*.jpg | sort -V | xargs -I {} cat {}"
            cmd_ffmpeg = " ".join(
                [
                    "ffmpeg",
                    "-f image2pipe",
                    f"-r {c.FRAME_RATE}",
                    "-vcodec mjpeg",
                    "-i -",
                    "-vcodec libx264",
                    vid_filename,
                ]
            )

            cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
            os.system(cmd)
            print("(done)")

            cmd = "rm -f /tmp/fuzzerdata-drivefuzz/front-*.jpg"
            os.system(cmd)

            print("Saving top camera video", end=" ")

            vid_filename = "/tmp/fuzzerdata-drivefuzz/top.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)

            cmd_cat = (
                "ls /tmp/fuzzerdata-drivefuzz/top-*.jpg | sort -V | xargs -I {} cat {}"
            )
            cmd_ffmpeg = " ".join(
                [
                    "ffmpeg",
                    "-f image2pipe",
                    f"-r {c.FRAME_RATE}",
                    "-vcodec mjpeg",
                    "-i -",
                    "-vcodec libx264",
                    vid_filename,
                ]
            )

            cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
            os.system(cmd)
            print("(done)")

            cmd = "rm -f /tmp/fuzzerdata-drivefuzz/top-*.jpg"
            os.system(cmd)
        if agent_wrapper is not None:
            agent_wrapper.cleanup()
        if agent is not None:
            agent.destroy()
        destroy_commands = []
        for v in actor_vehicles:
            destroy_commands.append(carla.command.DestroyActor(v))
        for w in actor_walkers:
            destroy_commands.append(carla.command.DestroyActor(w))
        for f in actor_frictions:
            destroy_commands.append(carla.command.DestroyActor(f))
        for s in sensors:
            destroy_commands.append(carla.command.DestroyActor(s))
        if player is not None:
            destroy_commands.append(carla.command.DestroyActor(player))
        client.apply_batch(destroy_commands)
        world.tick()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        tm.set_synchronous_mode(False)
        state.stop_simulate_time = time.time()
        try:
            dill.dump(state, open("state.pick", "wb"))
        except Exception as e_dill:
            print(e_dill)
        try:
            dill.dump(retval, open("retval.pick", "wb"))
        except Exception as e_dill:
            print(e_dill)


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


def timestamp_pickle(timestamp):
    return carla.Timestamp, (
        timestamp.frame,
        timestamp.elapsed_seconds,
        timestamp.delta_seconds,
        timestamp.platform_timestamp,
    )


if __name__ == "__main__":
    print("executor.py start")
    copyreg.pickle(carla.Location, location_pickle)
    copyreg.pickle(carla.Rotation, rotation_pickle)
    copyreg.pickle(carla.Transform, transform_pickle)
    copyreg.pickle(carla.Timestamp, timestamp_pickle)
    param = dill.load(open("param.pick", "rb"))
    simulate(*param)
    print("executor.py end")
