#!/usr/bin/env python3
########################
import coverage

cov = coverage.Coverage()
cov.start()
########################
import copyreg
import os
import random
import shutil
import subprocess
import sys
import re
import signal
from types import SimpleNamespace

import time
import math
import traceback
import cv2
import glob
import logging
import constants as c

import carla
from carla import VehicleLightState as vls

from oracle import AllOracle
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent  #

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


def try_tick(world):
    try:
        fp = world.tick()
        return fp
    except:
        # print("tick 丢失，重新发送")
        time.sleep(2)
        try_tick(world)


def _on_collision(event, state):
    # print("COLLISION:", event)

    if state.stop_scenario_flag:
        return

    if event.other_actor.type_id != "static.road":
        # do not count collision while spawning ego vehicle (hard drop)
        state.crashed = True
        state.collision_details.append((event.timestamp, event.transform))


def _on_invasion(event, state):
    if state.stop_scenario_flag:
        return

    crossed_lanes = event.crossed_lane_markings
    for crossed_lane in crossed_lanes:
        if crossed_lane.lane_change == carla.LaneChange.NONE:
            # print("LANE INVASION:", event)
            state.laneinvaded = True
            temp_event = SimpleNamespace()
            temp_event.frame = event.frame
            temp_event.timestamp = event.timestamp
            temp_event.transform = event.transform
            state.laneinvasion_event.append(temp_event)
            state.laneinvasion_details.append((event.timestamp, event.transform))


def _on_front_camera_capture(path, image):
    image.save_to_disk(f"{path}/front-{image.frame}.jpg")


def _on_top_camera_capture(path, image):
    image.save_to_disk(f"{path}/top-{image.frame}.jpg")


def get_trajectory_and_min_distance(
    world, actor_id_list, traject_dict, state, player_loc, type=None
):
    all_actor = world.get_actors(actor_id_list)
    for actor in all_actor:
        if actor is not None:
            actor_id = actor.id
            if actor_id not in traject_dict:
                traject_dict[actor_id] = {}
                if type is not None:
                    traject_dict[actor_id]["type"] = type
                traject_dict[actor_id]["trajectory"] = []
            location = actor.get_transform().location
            traject_dict[actor_id]["trajectory"].append(
                (location.x, location.y, location.z)
            )

            # compute the distance to the player

            dist = player_loc.distance(location)
            if dist < state.min_dist:
                state.min_dist = dist
    return traject_dict


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


def generate_and_delete_mp4(
    folder_path, file_pattern, output_path_name, frame_rate=20, debug=False
):
    files = sorted(
        glob.glob(os.path.join(folder_path, file_pattern)),
        key=lambda x: int(x.split("-")[-1].split(".")[0]),
    )

    if len(files) == 0:
        print(f"No matching files found for pattern: {file_pattern}")
        return

    img = cv2.imread(files[0])
    height, width, _ = img.shape
    output_path = os.path.join(folder_path, output_path_name)
    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), frame_rate, (width, height)
    )

    for file in files:
        img = cv2.imread(file)
        video.write(img)

    video.release()
    if debug:
        print(f"MP4 video generated successfully: {output_path}")

    for file in files:
        os.remove(file)
        if debug:
            print(f"Deleted: {file}")


# def connect(conf):
#     global client

#     client = carla.Client(conf.sim_host, conf.sim_port)
#     print(conf.sim_host, conf.sim_port)
#     client.set_timeout(10.0)
#     try:
#         client.get_server_version()
#     except Exception as e:
#         print("[-] Error: Check client connection.")
#         sys.exit(-1)
#     if conf.debug:
#         print("Connected to:", client)

#     return client


def switch_map(conf, town):
    """
    Switch map in the simulator and retrieve legitimate waypoints (a list of
    carla.Transform objects) in advance.
    """
    global client
    global list_spawn_points
    global town_map
    print("[*] Switching town to {} (slow)".format(town))
    carla_port_docker = os.environ.get("CARLA_PORT_DOCKER", 4000)
    try:
        client = carla.Client(conf.sim_host, int(carla_port_docker))
        client.set_timeout(10)
        world = client.get_world()
        # if world.get_map().name != town: # force load every time
        
        client.set_timeout(20)  # Handle sluggish loading bug
        if str(town).startswith("Town"):
            client.load_world(str(town))
        else:
            client.load_world("Town0" + str(town))  # e.g., "/Game/Carla/Maps/Town01"

        if conf.debug:
            print("[+] Switched")
        client.set_timeout(10.0)

        town_map = world.get_map()
        list_spawn_points = town_map.get_spawn_points()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE  # FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

    except Exception as e:
        print("[-] Error:", e)
        if conf.agent_type == c.AUTOWARE_0903:
            carla_v = "0.9.14"
        else:
            carla_v = "0.9.13"
        subprocess.Popen(
            ["./run_carla_v2.sh", "-p", str(carla_port_docker), "-v", carla_v, "-d"]
        )
        time.sleep(10)
        return switch_map(conf, town)


def get_set_traffic_light(
    vehicle, red_time=10, green_time=10, yellow_time=1, set_state=None
):
    state_dict = {
        "red": carla.TrafficLightState.Red,
        "green": carla.TrafficLightState.Green,
        "yellow": carla.TrafficLightState.Yellow,
    }
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        traffic_light.set_green_time(10)
        traffic_light.set_red_time(10)
        traffic_light.set_yellow_time(1)
        if set_state is not None:
            if traffic_light.get_state() == state_dict[set_state]:
                pass
            else:
                traffic_light.set_state(state_dict[set_state])


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
        client.set_timeout(10)
        if str(town).startswith("Town0"):
            client.load_world(str(town))
        else:
            client.load_world("Town0" + str(town))
        print(f"Loaded map Town0{str(town)}")
        tm = client.get_trafficmanager(int(carla_port_docker) + 2025)
        tm.set_synchronous_mode(True)
        print("TM_CLIENT:", tm, tm.get_port())
    except Exception as carla_connect_error:
        print(carla_connect_error)
        subprocess.Popen(
            ["./run_carla_v2.sh", "-p", str(carla_port_docker), "-v", carla_v, "-d"]
        )
        time.sleep(10)
        return simulate(
            conf, state, town, sp, wp, weather_dict, frictions_list, actors_list
        )
    if os.path.exists("/tmp/fuzzerdata-avfuzzer"):
        shutil.rmtree("/tmp/fuzzerdata-avfuzzer")
    os.mkdir("/tmp/fuzzerdata-avfuzzer")
    #####################
    state.location_recorder = []
    state.running_red_light_details = []
    state.stuck_details = []
    state.speeding_details = []
    state.collision_details = []
    state.laneinvasion_details = []
    state.start_simulate_time = time.time()
    state.stop_scenario_flag = False
    #####################
    all_oracle = None
    #####################
    retval = 0
    agent = None
    try:
        world = client.get_world()
        if conf.debug:
            print("[debug] world:", world)

        town_map = world.get_map()
        if conf.debug:
            print("[debug] map:", town_map)

        blueprint_library = world.get_blueprint_library()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE  # FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

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
            # autoware_2409_utils.zero_counters()
        frame_id = world.tick()
        init_frame_id = frame_id
        frame_0 = frame_id
        start_time = time.time()
        clock = pygame.time.Clock()

        # set weather
        if isinstance(weather_dict, dict):
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

        elif isinstance(weather_dict, str):
            names = [
                name
                for name in dir(carla.WeatherParameters)
                if re.match("[A-Z].+", name)
            ]
            weathers = {x: getattr(carla.WeatherParameters, x) for x in names}
            if "_night_time" not in weather_dict:
                weather = weathers[weather_dict]
            else:
                weather_name = weather_dict.replace("_night_time", "")
                weather = weathers[weather_name]
                weather.sun_altitude_angle = -90.0
            state.weather = {
                "cloud": weather.cloudiness,
                "rain": weather.precipitation,
                "puddle": weather.precipitation_deposits,
                "wetness": weather.wetness,
                "wind": weather.wind_intensity,
                "fog": weather.fog_density,
                "angle": weather.sun_azimuth_angle,
                "altitude": weather.sun_altitude_angle,
            }
            world.set_weather(weather)

        sensors = []
        actor_vehicles_liner = []
        actor_vehicles_immobile = []
        actor_vehicles_autopilot = []
        actor_vehicles_maneuver = []
        actor_walkers_liner = []
        actor_walkers_immobile = []
        actor_walkers_autopilot = []
        all_actor_id = []
        actor_frictions = []
        ros_pid = 0
        world.tick()  # sync once with simulator

        # spawn player
        # how ScenarioFuzz spawns a player vehicle depends on
        # the autonomous driving agent
        player_bp = blueprint_library.filter("mercedes")[0]
        # player_bp.set_attribute("role_name", "ego")
        player = None

        goal_loc = wp.location
        goal_rot = wp.rotation

        # mark goal position,3Dbox
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
            try:
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

                camera_front.listen(
                    lambda image: _on_front_camera_capture(conf.cache_dir, image)
                )

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

                camera_top.listen(
                    lambda image: _on_top_camera_capture(conf.cache_dir, image)
                )
                sensors.append(camera_top)
            except:
                print("[sensor error]: front and top video camera spawn not successed!")
                retval = -1
                return

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

        # -------------------------- spawn actors,new way by wt------------------------------------------
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor
        SetActorSpeed = carla.command.ApplyTargetVelocity
        Delete_actor = carla.command.DestroyActor
        """
        new_actor = {
                "type": actor_type,
                "nav_type": nav_type,
                "spawn_point": spawn_point,
                "dest_point": dest_point,
                "speed": speed,
                "maneuvers": maneuvers,
                "bp_id":name,
                'color':color,
                'dp_time':dp_time
            }
        """
        light_state = vls.NONE
        if weather.sun_altitude_angle < 0:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam
        batch = []
        actor_id_list = []
        for index, actor in enumerate(actors_list):

            # create a list to keep track of static actors to delete
            actor_sp = get_carla_transform(actor["spawn_point"])

            actor_type = actor["type"]
            actor_nav_type = actor["nav_type"]
            actor_bp_id = actor["bp_id"]
            actor_color = actor["color"]

            command = None
            if actor_type == c.VEHICLE:
                vehicle_bp = blueprint_library.find(actor_bp_id)
                if vehicle_bp.has_attribute("color"):
                    vehicle_bp.set_attribute(
                        "color", f"{actor_color[0]},{actor_color[1]},{actor_color[2]}"
                    )
                vehicle_bp.set_attribute(
                    "role_name", "{}_vehicle".format(c.NAVTYPE_NAMES[actor_type])
                )
                command = SpawnActor(vehicle_bp, actor_sp).then(
                    SetVehicleLightState(FutureActor, light_state)
                )
                if actor_nav_type == c.AUTOPILOT:
                    command = command.then(
                        SetAutopilot(FutureActor, True, tm.get_port())
                    )

            elif actor_type == c.WALKER:
                walker_bp = blueprint_library.find(actor_bp_id)
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                walker_bp.set_attribute(
                    "role_name", "{}_walker".format(c.NAVTYPE_NAMES[actor_type])
                )
                command = SpawnActor(walker_bp, actor_sp)

            if command is not None:
                batch.append(command)

        act_w = 0
        act_v = 0
        try:
            responses = client.apply_batch_sync(batch, True)
        except RuntimeError as e:
            print(f"[-] Error occurred during apply_batch_sync: {e}")
            retval = -1
            return  # trap to finally

        for index, response in enumerate(responses):
            actor_type = actors_list[index]["type"]
            nav_type = c.NAVTYPE_NAMES[actors_list[index]["nav_type"]]
            bp_id = actors_list[index]["bp_id"]
            sp_loc = actors_list[index]["spawn_point"]
            if response.error:
                print("[-] Failed spawning {} {} at {}".format(nav_type, bp_id, sp_loc))
                state.spawn_failed = True
                state.spawn_failed_object = actor
                retval = -1
                return  # trap to finally
            else:
                print("[+] New {} {} spawing at {} .".format(nav_type, bp_id, sp_loc))
                actor_id_list.append(response.actor_id)
                if actor_type == c.VEHICLE:
                    act_v += 1
                elif actor_type == c.WALKER:
                    act_w += 1

        # -------------------------------------------------------spawn end---------------------------------------------------------------

        # handle actor missions after Autoware's goal is published

        # ---------------------------------------------set other vehicle and walkers dynamic state-------------------------------------------------

        walker_controller_bp = world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        actor_instance_list = world.get_actors(actor_id_list)
        aw_batch = []
        for index, (actor_id, actor) in enumerate(zip(actor_id_list, actors_list)):

            actor_sp = get_carla_transform(actor["spawn_point"])
            actor_dp_time = actor["dp_time"]

            if actor["type"] == c.VEHICLE:

                actor_vehicle = actor_instance_list[index]
                actor_vehicle.set_simulate_physics(True)
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * actor["speed"])
                    actor_vehicles_liner.append(actor_id)

                if actor["nav_type"] == c.IMMOBILE:
                    actor_vehicles_immobile.append([actor_id, actor_dp_time])

                if actor["nav_type"] == c.AUTOPILOT:
                    actor_vehicles_autopilot.append(actor_id)

                elif actor["nav_type"] == c.MANEUVER:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * 0)
                    actor_vehicles_maneuver.append([actor_id, index])

            elif actor["type"] == c.WALKER:  # walker

                actor_walker = actor_instance_list[index]
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    controller_walker = carla.WalkerControl()
                    controller_walker.direction = forward_vec
                    controller_walker.speed = actor["speed"]
                    actor_walker.apply_control(controller_walker)
                    actor_walkers_liner.append(actor_id)
                elif actor["nav_type"] == c.AUTOPILOT:
                    aw_batch.append(
                        SpawnActor(
                            walker_controller_bp, carla.Transform(), actor_walker
                        )
                    )
                    actor_walkers_autopilot.append([actor_id, index])

                elif actor["nav_type"] == c.IMMOBILE:
                    actor_walkers_immobile.append([actor_id, actor_dp_time])

        if len(aw_batch) != 0:
            results = client.apply_batch_sync(aw_batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    actor_walkers_autopilot[i].append(results[i].actor_id)

            actor_walker_autopilot_control_list = [
                a[2] for a in actor_walkers_autopilot
            ]
            awc_instance_list = world.get_actors(actor_walker_autopilot_control_list)

            for index, aw in enumerate(actor_walkers_autopilot):
                actor_index = aw[1]
                actor = actors_list[actor_index]
                actor_wp = get_carla_transform(actor["dest_point"])
                actor_speed = actor["speed"]

                try:
                    # 启动行人
                    awc_instance_list[index].start()
                    # 设置行人前往随机点
                    awc_instance_list[index].go_to_location(actor_wp.location)
                    # 设置最大速度
                    awc_instance_list[index].set_max_speed(actor_speed)
                except Exception as e:
                    logging.error(f"设置行人目的地和速度时出错：{e}")

        world.tick()
        # ---------------------------------dynamic for other vehicle and walkers end!---------------------------------------

        elapsed_time = 0
        start_time = time.time()

        yaw = sp.rotation.yaw

        player_loc = player.get_transform().location
        init_x = player_loc.x
        init_y = player_loc.y

        # SIMULATION LOOP FOR AUTOWARE and BasicAgent
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGSEGV, state.sig_handler)
        signal.signal(signal.SIGABRT, state.sig_handler)

        try:
            # time logging

            exec_scenario_finish_time = datetime.now()
            state.exec_scenario_finish_time = exec_scenario_finish_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            # actual monitoring of the driving simulation begins here
            snapshot0 = world.get_snapshot()
            first_frame_id = snapshot0.frame
            first_sim_time = snapshot0.timestamp.elapsed_seconds
            if conf.agent_type == c.INTERFUSER:
                GameTime_interfuser.on_carla_tick(snapshot0.timestamp)
            elif conf.agent_type == c.LMDRIVE:
                GameTime_lmdriver.on_carla_tick(snapshot0.timestamp)

            last_frame_id = first_frame_id
            last_sim_time = first_sim_time
            state.first_frame_id = first_frame_id
            state.sim_start_time = snapshot0.timestamp.platform_timestamp
            state.num_frames = 0
            state.elapsed_time = 0
            state._last_frame = snapshot0.frame
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
            start_loop_time = time.time()
            all_oracle = AllOracle(player, world, conf.out_dir)
            while time.time() - start_loop_time < 10 * 60:
                # Use sampling frequency of FPS*2 for precision!
                clock.tick(c.FRAME_RATE * 2)

                # Carla agents are running in synchronous mode,
                # so we need to send ticks. Not needed for Autoware
                # check each actor in the dictionary
                if (state.num_frames) % (c.FRAME_RATE * 5) == 0:
                    batch = []
                    remove_aw = []
                    if len(actor_walkers_autopilot) != 0:
                        for index, awc in enumerate(actor_walkers_autopilot):
                            aw_id = awc[0]
                            ac_id = awc[2]
                            a_id = awc[1]
                            actor_aw = world.get_actor(aw_id)
                            if actor_aw is not None:
                                actor = actors_list[a_id]
                                actor_wp = get_carla_transform(actor["dest_point"])
                                actor_acw = world.get_actor(ac_id)
                                if (
                                    actor_acw is not None
                                    and actor_aw.get_location().distance(
                                        actor_wp.location
                                    )
                                    < 4.0
                                ):
                                    actor_acw.stop()
                                    batch.append(Delete_actor(aw_id))
                                    batch.append(Delete_actor(ac_id))
                                    print(
                                        f"autopilot walker name {aw_id} and controller {ac_id} disappear!"
                                    )
                                    remove_aw.append(awc)

                    remove_iv = []
                    if len(actor_vehicles_immobile) != 0:
                        for avi in actor_vehicles_immobile:
                            avi_i = avi[0]
                            avi_t = avi[1]
                            if state.elapsed_time > avi_t:
                                actor_avi = world.get_actor(avi_i)
                                if actor_avi is not None:
                                    batch.append(Delete_actor(avi_i))
                                    print(f"immobile vehicle name {avi_i} disappear!")
                                    remove_iv.append(avi)

                    remove_iw = []
                    if len(actor_walkers_immobile) != 0:
                        for awi in actor_walkers_immobile:
                            awi_i = awi[0]
                            awi_t = awi[1]
                            if state.elapsed_time > awi_t:
                                actor_awi = world.get_actor(awi_i)
                                if actor_awi is not None:
                                    batch.append(Delete_actor(awi_i))
                                    print(f"immobile walker name {awi_i} disappear!")
                                    remove_iw.append(awi)

                    if len(batch) != 0:
                        client.apply_batch_sync(batch, True)

                        if conf.agent_type == c.AUTOWARE:
                            world.tick()

                    # Remove deleted actors from the lists
                    for awc in remove_aw:
                        actor_walkers_autopilot.remove(awc)

                    for avi in remove_iv:
                        actor_vehicles_immobile.remove(avi)

                    for awi in remove_iw:
                        actor_walkers_immobile.remove(awi)

                # --------------------------------del_list_end-------------------------------------

                if (
                    conf.agent_type == c.BASIC
                    or conf.agent_type == c.BEHAVIOR
                    or conf.agent_type == c.INTERFUSER
                    or conf.agent_type == c.LMDRIVE
                ):
                    world.tick()

                snapshot = world.get_snapshot()
                cur_frame_id = snapshot.frame  ##
                cur_sim_time = snapshot.timestamp.elapsed_seconds
                if conf.agent_type == c.INTERFUSER:
                    GameTime_interfuser.on_carla_tick(snapshot.timestamp)
                elif conf.agent_type == c.LMDRIVE:
                    GameTime_lmdriver.on_carla_tick(snapshot.timestamp)
                state._last_frame = snapshot.frame
                if cur_frame_id <= last_frame_id:
                    # skip if we got the same frame data as last
                    continue

                last_frame_id = cur_frame_id  # update last
                last_sim_time = cur_sim_time
                state.num_frames = cur_frame_id - first_frame_id
                state.elapsed_time = cur_sim_time - first_sim_time
                player_transform = player.get_transform()
                player_loc = player_transform.location
                player_rot = player_transform.rotation

                try:
                    get_set_traffic_light(player)
                except:
                    pass
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
                if "ego_car" not in state.object_trajectory:
                    state.object_trajectory["ego_car"] = {}
                    state.object_trajectory["ego_car"]["trajectory"] = []

                state.object_trajectory["ego_car"]["trajectory"].append(
                    (player_loc.x, player_loc.y, player_loc.z)
                )
                # print("(%.2f,%.2f)>(%.2f,%.2f)>(%.2f,%.2f) %.2f m left, %.2f/%d km/h   " % (
                #     sp.location.x, sp.location.y, player_loc.x,
                #     player_loc.y, goal_loc.x, goal_loc.y,
                #     player_loc.distance(goal_loc),
                #     speed, speed_limit), end="")
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
                    # for carla agents, we should apply controls ourselves
                    # XXX: check and resolve BehaviorAgent's run_step issue of
                    # not being able to get adjacent waypoints

                    control = agent.run_step(debug=conf.debug)
                    player.apply_control(control)

                elif conf.agent_type == c.BEHAVIOR:

                    agent._update_information()
                    agent.get_local_planner().set_speed(speed_limit)

                    control = agent.run_step(debug=conf.debug)
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
                # Yaw range is -180 ~ 180. When vehicle's yaw is oscillating
                # b/w -179 and 179, yaw_diff can be messed up even if the
                # diff is very small. Assuming that it's unrealistic that
                # a vehicle turns more than 180 degrees in under 1/20 seconds,
                # just round the diff around 360.
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                elif yaw_diff < -180:
                    yaw_diff = 360 + yaw_diff

                yaw_rate = yaw_diff * c.FRAME_RATE
                state.yaw_rate_list.append(yaw_rate)
                yaw = current_yaw

                # Get the lateral speed
                player_right_vec = player_rot.get_right_vector()

                # [Note]
                # Lateral velocity is a scalar projection of velocity vector.
                # A: velocity vector.
                # B: right vector. B is a unit vector, thus |B| = 1
                # lat_speed = |A| * cos(theta)
                # As dot_product(A, B) = |A| * |B| * cos(theta),
                # lat_speed = dot_product(A, B) / |B|
                # Given that |B| is always 1,
                # we get lat_speed = dot_product(A, B), which is equivalent to
                # lat_speed = vel.x * right_vel.x + vel.y * right_vel.y

                lat_speed = abs(vel.x * player_right_vec.x + vel.y * player_right_vec.y)
                lat_speed *= 3.6  # m/s to km/h
                state.lat_speed_list.append(lat_speed)

                player_fwd_vec = player_rot.get_forward_vector()
                lon_speed = abs(vel.x * player_fwd_vec.x + vel.y * player_fwd_vec.y)
                lon_speed *= 3.6
                state.lon_speed_list.append(lon_speed)

                # Handle actor maneuvers
                for actor_id, index in actor_vehicles_maneuver:
                    actor_bp = world.get_actor(actor_id)
                    actor = actors_list[index]
                    maneuvers = actor["maneuvers"]

                    maneuver_id = int(state.num_frames / c.FRAMES_PER_TIMESTEP)
                    if maneuver_id < 5:
                        maneuver = maneuvers[maneuver_id]

                        if maneuver[2] == 0:
                            # print(f"\nPerforming maneuver #{maneuver_id} at frame {state.num_frames}")
                            # mark as done
                            maneuver[2] = state.num_frames

                            # retrieve the actual actor vehicle object
                            # there is only one actor in Trajectory mode
                            actor_vehicle = actor_bp

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

                # -----------------------------store other actors' information and check min dist --------------------------
                state.object_trajectory.setdefault("walkers", {})
                state.object_trajectory.setdefault("vehicles", {})

                state.object_trajectory["walkers"] = get_trajectory_and_min_distance(
                    world,
                    actor_walkers_liner,
                    state.object_trajectory["walkers"],
                    state,
                    player_loc,
                    "linear",
                )
                state.object_trajectory["walkers"] = get_trajectory_and_min_distance(
                    world,
                    [a[0] for a in actor_walkers_immobile],
                    state.object_trajectory["walkers"],
                    state,
                    player_loc,
                    "immobile",
                )
                state.object_trajectory["walkers"] = get_trajectory_and_min_distance(
                    world,
                    [a[0] for a in actor_walkers_autopilot],
                    state.object_trajectory["walkers"],
                    state,
                    player_loc,
                    "autopilot",
                )

                state.object_trajectory["vehicles"] = get_trajectory_and_min_distance(
                    world,
                    actor_vehicles_liner,
                    state.object_trajectory["vehicles"],
                    state,
                    player_loc,
                    "linear",
                )
                state.object_trajectory["vehicles"] = get_trajectory_and_min_distance(
                    world,
                    [a[0] for a in actor_vehicles_immobile],
                    state.object_trajectory["vehicles"],
                    state,
                    player_loc,
                    "immobile",
                )
                state.object_trajectory["vehicles"] = get_trajectory_and_min_distance(
                    world,
                    actor_vehicles_autopilot,
                    state.object_trajectory["vehicles"],
                    state,
                    player_loc,
                    "autopilot",
                )
                state.object_trajectory["vehicles"] = get_trajectory_and_min_distance(
                    world,
                    [a[0] for a in actor_vehicles_maneuver],
                    state.object_trajectory["vehicles"],
                    state,
                    player_loc,
                    "maneuver",
                )

                # ------------------------------end check-----------------------------------------------------

                # --------------------if state.fitness function exists------------------------------------
                if state.fitness_cal_object is not None:
                    state.fitness_cal_object.extract_from_world(player, world, goal_loc)
                dist_to_goal = player_loc.distance(goal_loc)
                d_min_end = c.MIN_DIST_TO_GOAL
                waypoint_queue = []
                if conf.agent_type == c.BASIC:
                    if hasattr(agent, "done") and agent.done():
                        print("\n[*] (BasicAgent) Reached the destination")

                        if dist_to_goal > d_min_end and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal

                        break

                elif conf.agent_type == c.BEHAVIOR:
                    lp = agent.get_local_planner()
                    if len(lp._waypoints_queue) == 0:
                        print("\n[*] (BehaviorAgent) Reached the destination")

                        if dist_to_goal > d_min_end and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal

                        break
                elif conf.agent_type == c.INTERFUSER:
                    waypoint_queue = agent._route_planner.route
                    if len(agent._route_planner.route) == 0:
                        print("\n[*] (InterFuser) Reached the destination")

                        if dist_to_goal > d_min_end and state.num_frames > 300:
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
                if dist_to_goal <= d_min_end:
                    print("\n[*] (Carla heuristic) Reached the destination")
                    retval = 666
                    break
                all_oracle.update(snapshot, goal_loc, waypoint_queue)
                # Check speeding
                if conf.check_dict["speed"]:
                    # allow T seconds to slow down if speed limit suddenly
                    # decreases
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
                    if state.stuck_duration > (60 * c.FRAME_RATE):
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

        # jump to finally
        return

    except Exception as e:
        retval = -1
        print("[-] Runtime error:")
        traceback.print_exc()
    finally:
        if conf.agent_type == c.AUTOWARE_0903:
            autoware_2409_utils.monitor_routing_event.set()
            autoware_2409_utils.stop_autoware()
            autoware_2409_utils.stop_bridge()
        if all_oracle:
            all_oracle.save_2_file()
            # if cov:
            #     cov.stop()
            #     cov.save()
            #     cov.xml_report(
            #         outfile=os.path.join(
            #             conf.out_dir,
            #             "oracles",
            #             str(all_oracle.start_sim_time),
            #             "coverage.xml",
            #         )
            #     )
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
        state.stop_simulate_time = time.time()
        state.stop_scenario_flag = True
        if (
            conf.agent_type == c.BASIC
            or conf.agent_type == c.BEHAVIOR
            or conf.agent_type == c.INTERFUSER
            or conf.agent_type == c.LMDRIVE
            or conf.agent_type == c.AUTOWARE_0903
        ):
            # assemble images into an mp4 container
            # remove jpg files
            folder_path = conf.cache_dir  # 替换为你的文件夹路径
            output_path_front = "front.mp4"  # 输出 front 类型视频文件路径
            output_path_top = "top.mp4"  # 输出 top 类型视频文件路径
            frame_rate = c.FRAME_RATE  # 帧率

            vid_filename = f"{conf.cache_dir}/front.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)

            vid_filename = f"{conf.cache_dir}/top.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)

            print("Saving front camera video", end=" ")
            generate_and_delete_mp4(
                folder_path, "front-*.jpg", output_path_front, frame_rate
            )
            print("(done)")
            time.sleep(2)

            print("Saving top camera video", end=" ")
            generate_and_delete_mp4(
                folder_path, "top-*.jpg", output_path_top, frame_rate
            )
            print("(done)")

        destroy_commands = []

        if len(all_actor_id) != 0:
            for i in all_actor_id:
                destroy_commands.append(carla.command.DestroyActor(i))
        if len(actor_frictions) != 0:
            for f in actor_frictions:
                destroy_commands.append(carla.command.DestroyActor(f))
        if len(sensors) != 0:
            for s in sensors:
                destroy_commands.append(carla.command.DestroyActor(s))

        if player is not None:
            destroy_commands.append(carla.command.DestroyActor(player))
        client.apply_batch(destroy_commands)

        return retval, state


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
    copyreg.pickle(carla.Location, location_pickle)
    copyreg.pickle(carla.Rotation, rotation_pickle)
    copyreg.pickle(carla.Transform, transform_pickle)
    copyreg.pickle(carla.Timestamp, timestamp_pickle)
    param = dill.load(open("param.pick", "rb"))
    retval, state = simulate(*param)
    dill.dump(state, open("state.pick", "wb"))
    dill.dump(retval, open("retval.pick", "wb"))
    print(f"Finish executor.py with {retval} and {state}")
