import os
import json
import time, datetime
import shutil
import math

import numpy as np

import config
from driving_quality import *
import random

import carla

import executor
import constants as c
import dill


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
        carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2])
    )

    return t


def get_valid_xy_range(town):
    try:
        with open(os.path.join("town_info", town + ".json")) as fp:
            town_data = json.load(fp)
    except:
        return (-999, 999, -999, 999)

    x_list = []
    y_list = []
    for coord in town_data:
        x_list.append(town_data[coord][0])
        y_list.append(town_data[coord][1])

    return (min(x_list), max(x_list), min(y_list), max(y_list))


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    # Copied from
    # https://github.com/davheld/tf/blob/master/src/tf/transformations.py#L1100

    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

    _NEXT_AXIS = [1, 2, 0, 1]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def node2wp(node, map):
    return map.get_waypoint(location=carla.Location(node[0], node[1], node[2]))


class TestScenario:
    seed_data = {}
    town = None
    # sp = None
    # wp = None
    weather = {}
    actors = []
    puddles = []
    # oracle_state = None
    # client = None
    # tm = None
    # list_spawn_points = None
    # quota = 0
    # last_deduction = 0
    driving_quality_score = None
    found_error = False
    event_dict = {
        "crash": False,
        "stuck": False,
        "lane_invasion": False,
        "red": False,
        "speeding": False,
        "other": False,
        "other_error_val": False
    }

    def __init__(self, conf):
        """
        When initializing, perform dry run and get the oracle state
        """
        self.conf = conf

        self.weather_by_name = self.conf.weather_by_name

        if not self.weather_by_name:
            self.weather = {}
            self.weather["cloud"] = 0
            self.weather["rain"] = 0
            self.weather["puddle"] = 0
            self.weather["wind"] = 0
            self.weather["fog"] = 0
            self.weather["wetness"] = 0
            self.weather["angle"] = 0
            self.weather["altitude"] = 90
        else:
            self.weather = 'ClearNoon'

        self.actors = []
        self.puddles = []
        self.driving_quality_score = 0
        self.found_error = False

        self.log_filename = None

        if conf.town is not None:
            self.town = conf.town

        executor.switch_map(conf, self.town)

        print("[+] test case initialized")

    def set_ego_wp_sp(self, sp, wp, specific_waypoints=False):

        def get_tf_sp(node):
            map = executor.town_map
            wp = node2wp(node, map)
            wp_tf = wp.transform
            node_tf = carla.Transform(
                carla.Location(x=wp_tf.location.x, y=wp_tf.location.y, z=max(2, wp_tf.location.z + 1)),
                carla.Rotation(roll=0, pitch=0, yaw=wp_tf.rotation.yaw)
            )
            return node_tf

        def get_tf_wp(node):
            map = executor.town_map
            wp = node2wp(node, map)
            wp_tf = wp.transform
            node_tf = carla.Transform(
                carla.Location(x=wp_tf.location.x, y=wp_tf.location.y, z=max(2, wp_tf.location.z + 1)),
                carla.Rotation(roll=0, pitch=0, yaw=wp_tf.rotation.yaw)
            )
            return node_tf

        def transform_carla_point(point):
            location = carla.Location(x=point[0][0], y=point[0][1], z=point[0][2])
            rotation = carla.Rotation(roll=point[1][0], pitch=point[1][1], yaw=point[1][2])
            return carla.Transform(location, rotation)

        self.ego_sp = get_tf_sp(sp) if not specific_waypoints else transform_carla_point(sp)
        self.ego_wp = get_tf_wp(wp) if not specific_waypoints else transform_carla_point(wp)

    def get_seed_sp_transform(self, seed):
        sp = carla.Transform(
            carla.Location(seed["sp_x"], seed["sp_y"], seed["sp_z"]),
            carla.Rotation(roll=seed["sp_roll"], pitch=seed["sp_pitch"], yaw=seed["sp_yaw"])
        )

        return sp

    def get_seed_wp_transform(self, seed):
        wp = carla.Transform(
            carla.Location(seed["wp_x"], seed["wp_y"], seed["wp_z"]),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=seed["wp_yaw"])
        )

        return wp

    def get_distance_from_player(self, location):
        sp = self.get_seed_sp_transform(self.seed_data)
        return location.distance(sp.location)

    def ego_front_side_walk_location(self, sp):

        if isinstance(sp, carla.Transform):
            sp = sp.location
            wp1 = executor.town_map.get_waypoint(sp, lane_type=(carla.LaneType.Sidewalk))
        elif isinstance(sp, carla.Location):
            wp1 = executor.town_map.get_waypoint(sp, lane_type=(carla.LaneType.Sidewalk))
        else:
            sp = get_carla_transform(sp)
            wp1 = executor.town_map.get_waypoint(sp.location, lane_type=(carla.LaneType.Sidewalk))
        return wp1

    def get_along_waypoint(self, waypoint):
        waypoint = waypoint.next(10)  # next(self, distance)
        if len(waypoint) > 0:
            waypoint = waypoint[0]  # select the first wp
            waypoint = executor.town_map.get_waypoint(waypoint.transform.location, lane_type=(carla.LaneType.Sidewalk))
        else:
            return -1
        return waypoint

    def get_locations_sidebike(self, wp_opposite, people_ids):
        # locate peoples 
        # even in the same direction
        transforms = []

        wp_ego = self.ego_front_side_walk_location(self.ego_sp)

        wp_opposite = self.ego_front_side_walk_location(wp_opposite)

        for i in range(0, people_ids):
            if i % 2 == 0:
                if wp_ego != -1:
                    wp_ego = self.get_along_waypoint(wp_ego)
                    if wp_ego != -1:
                        # 使用字典代替 Location 和 Transform 类型
                        x, y, z = wp_ego.transform.location.x, wp_ego.transform.location.y, wp_ego.transform.location.z
                        roll, pitch, yaw = wp_ego.transform.rotation.roll, wp_ego.transform.rotation.pitch, wp_ego.transform.rotation.yaw
                        transform = {'x': x, 'y': y, 'z': z + 0.8000, 'roll': roll, 'pitch': pitch, 'yaw': yaw}
                        transforms.append(transform)
            else:
                if wp_opposite != -1:
                    wp_opposite = self.get_along_waypoint(wp_opposite)
                    if wp_opposite != -1:
                        x, y, z = wp_opposite.transform.location.x, wp_opposite.transform.location.y, wp_opposite.transform.location.z
                        roll, pitch, yaw = wp_opposite.transform.rotation.roll, wp_opposite.transform.rotation.pitch, wp_opposite.transform.rotation.yaw
                        transform = {'x': x, 'y': y, 'z': z + 0.8000, 'roll': roll, 'pitch': pitch, 'yaw': yaw}
                        transforms.append(transform)

        return transforms

    def add_actor(self, actor_type, nav_type, sp, wp, speed, name, color=(255, 255, 255), dp_time=None, maneuvers=None,
                  specific_waypoints=False):
        """
        Mutator calls `ret = add_actor` with the mutated parameters
        until ret == 0.

        actor_type: int
        location: (float x, float y, float z)
        rotation: (float yaw, float roll, float pitch)
        speed: float

        1) check if the location is within the map
        2) check if the location is not preoccupied by other actors
        3) check if the actor's path collides with player's path
        return 0 iif 1), 2), 3) are satisfied.
        """

        def get_loc_rot(node):
            map = executor.town_map
            wp = node2wp(node, map)
            wp_tf = wp.transform
            loc = (wp_tf.location.x, wp_tf.location.y, max(2, wp_tf.location.z + 1))
            rot = (0, 0, wp_tf.rotation.yaw)
            return (loc, rot)

        spawn_point = get_loc_rot(sp) if not specific_waypoints else sp  # ((loc),(rot))

        # do validity checks
        if nav_type == c.LINEAR:
            dest_point = None
            dp_time = None
        if nav_type == c.IMMOBILE:
            dest_point = None
            speed = 0
            dp_time = dp_time
        elif nav_type == c.MANEUVER:

            dest_point = None
            speed = 0
            dp_time = None
            # [direction (-1: L, 0: Fwd, 1: R),
            #  velocity (m/s) if fwd / apex degree if lane change,
            #  frame_maneuver_performed]
            if maneuvers is None:
                maneuvers = [
                    [0, 0, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                ]
                i = random.randint(0, 4)
                direction = random.randint(-1, 1)

                if direction == 0:
                    speed = random.randint(0, 10)  # m/s
                    maneuvers[i] = [direction, speed, 0]
                else:
                    degree = random.randint(30, 60)  # deg
                    maneuvers[i] = [direction, degree, 0]

                # reset executed frame id
                for i in range(5):
                    maneuvers[i][2] = 0
            else:
                maneuvers = maneuvers
        elif nav_type == c.AUTOPILOT:
            if wp is not None:
                dest_point = get_loc_rot(wp) if not specific_waypoints else wp
            else:
                dest_point = None
            speed = 0
            dp_time = None
        new_actor = {
            "type": actor_type,
            "nav_type": nav_type,
            "spawn_point": spawn_point,
            "dest_point": dest_point,
            "speed": speed,
            "maneuvers": maneuvers,
            "bp_id": name,
            'color': color,
            'dp_time': dp_time
        }
        self.actors.append(new_actor)

        return 0

    def add_puddle(self, level, location, size):
        """
        Mutator calls `ret = add_friction` with the mutated parameters
        until ret == 0.

        level: float [0.0:1.0]
        location: (float x, float y, float z)
        size: (float xlen, float ylen, float zlen)

        1) check if the location is within the map
        2) check if the friction box lies on the player's path
        return 0 iff 1, and 2) are satisfied.
        """

        rotation = (0, 0, 0)
        spawn_point = (location, rotation)  # carla.Transform(location, carla.Rotation())

        new_puddle = {
            "level": level,
            "size": size,
            "spawn_point": spawn_point
        }

        self.puddles.append(new_puddle)

        return 0

    def dump_states(self, state, log_type):
        if self.conf.debug:
            print("[*] dumping {} data".format(log_type))

        state_dict = {}

        state_dict["fuzzing_start_time"] = self.conf.cur_time

        state_dict["campaign_cnt"] = state.campaign_cnt
        state_dict["cycle_cnt"] = state.cycle_cnt
        state_dict["mutation"] = state.mutation

        state_dict['campaign_start_time'] = self.conf.time_list['campaign_start_time']
        state_dict['cycle_start_time'] = self.conf.time_list['cycle_start_time']
        state_dict['mutation_start_time'] = self.conf.time_list['mutation_start_time']

        state_dict["exec_start_time"] = state.exec_start_time
        state_dict["exec_scenario_finish_time"] = state.exec_scenario_finish_time
        state_dict["exec_end_time"] = state.exec_end_time

        state_dict["determ_seed"] = self.conf.determ_seed
        state_dict["seed"] = self.seed_data
        if not self.weather_by_name:
            state_dict["weather"] = self.weather
        else:
            state_dict["weather"] = state.weather
        state_dict["autoware_cmd"] = state.autoware_cmd
        state_dict["autoware_goal"] = state.autoware_goal
        state_dict['map_name'] = self.town
        state_dict['ego_car'] = {
            'sp_x': self.ego_sp.location.x
            , 'sp_y': self.ego_sp.location.y
            , 'sp_z': self.ego_sp.location.z
            , 'sp_roll': self.ego_sp.rotation.roll
            , 'sp_pitch': self.ego_sp.rotation.pitch
            , 'sp_yaw': self.ego_sp.rotation.yaw
            , 'dp_x': self.ego_wp.location.x
            , 'dp_y': self.ego_wp.location.y
            , 'dp_z': self.ego_wp.location.z
        }
        actor_list = []
        for actor in self.actors:  # re-convert from carla.transform to xyz
            actor_dict = {
                "type": actor["type"],
                "nav_type": actor["nav_type"],
                "speed": actor["speed"],
            }
            if actor["spawn_point"] is not None:
                actor_dict["sp_x"] = actor["spawn_point"][0][0]
                actor_dict["sp_y"] = actor["spawn_point"][0][1]
                actor_dict["sp_z"] = actor["spawn_point"][0][2]
                actor_dict["sp_roll"] = actor["spawn_point"][1][0]
                actor_dict["sp_pitch"] = actor["spawn_point"][1][1]
                actor_dict["sp_yaw"] = actor["spawn_point"][1][2]

            if actor["dest_point"] is not None:
                actor_dict["dp_x"] = actor["dest_point"][0][0]
                actor_dict["dp_y"] = actor["dest_point"][0][1]
                actor_dict["dp_z"] = actor["dest_point"][0][2]
            actor_list.append(actor_dict)
        state_dict["actors"] = actor_list

        puddle_list = []
        for puddle in self.puddles:
            puddle_dict = {
                "level": puddle["level"],
                "sp_x": puddle["spawn_point"][0][0],
                "sp_y": puddle["spawn_point"][0][1],
                "sp_z": puddle["spawn_point"][0][2],
            }
            puddle_dict["size_x"] = puddle["size"][0]
            puddle_dict["size_y"] = puddle["size"][1]
            puddle_dict["size_z"] = puddle["size"][2]
            puddle_list.append(puddle_dict)
        state_dict["puddles"] = puddle_list

        state_dict["first_frame_id"] = state.first_frame_id
        state_dict["first_sim_elapsed_time"] = state.first_sim_elapsed_time
        state_dict["sim_start_time"] = state.sim_start_time
        state_dict["num_frames"] = state.num_frames
        state_dict["elapsed_time"] = state.elapsed_time

        state_dict["deductions"] = state.deductions

        vehicle_state_dict = {
            "speed": state.speed,
            "steer_wheel_angle": state.steer_angle_list,
            "yaw": state.yaw_list,
            "yaw_rate": state.yaw_rate_list,
            "lat_speed": state.lat_speed_list,
            "lon_speed": state.lon_speed_list,
            "min_dist": state.min_dist
        }
        state_dict["vehicle_states"] = vehicle_state_dict

        control_dict = {
            "throttle": state.cont_throttle,
            "brake": state.cont_brake,
            "steer": state.cont_steer
        }
        state_dict["control_cmds"] = control_dict

        event_dict = {
            "crash": state.crashed,
            "stuck": state.stuck,
            "lane_invasion": state.laneinvaded,
            "red": state.red_violation,
            "speeding": state.speeding,
            "other": state.other_error,
            "other_error_val": state.other_error_val
        }
        state_dict["events"] = event_dict
        events_inf = {}

        state_dict["events_inf"] = events_inf
        self.event_dict = event_dict
        config_dict = {
            "fps": c.FRAME_RATE,
            "max_dist_from_player": c.MAX_DIST_FROM_PLAYER,
            "min_dist_from_player": c.MIN_DIST_FROM_PLAYER,
            "abort_seconds": self.conf.timeout,
            "wait_autoware_num_topics": c.WAIT_AUTOWARE_NUM_TOPICS
        }
        state_dict["config"] = config_dict
        state_dict['object_trajectory'] = state.object_trajectory

        #####################################
        state_dict["location_recorder"] = state.location_recorder
        #####################################
        state_dict["laneinvasion_details"] = []
        for time_stamp, player_transform in state.laneinvasion_details:
            state_dict["laneinvasion_details"].append({
                "time_stamp": time_stamp,
                "x": player_transform.location.x,
                "y": player_transform.location.y,
                "z": player_transform.location.z,
            })
        state_dict["collision_details"] = []
        for time_stamp, player_transform in state.collision_details:
            state_dict["collision_details"].append({
                "time_stamp": time_stamp,
                "x": player_transform.location.x,
                "y": player_transform.location.y,
                "z": player_transform.location.z,
            })
        state_dict["running_red_light_details"] = []
        for time_stamp, player_transform in state.running_red_light_details:
            state_dict["running_red_light_details"].append({
                "time_stamp": time_stamp.elapsed_seconds,
                "x": player_transform.location.x,
                "y": player_transform.location.y,
                "z": player_transform.location.z,
            })
        state_dict["stuck_details"] = []
        for time_stamp, player_transform in state.stuck_details:
            state_dict["stuck_details"].append({
                "time_stamp": time_stamp.elapsed_seconds,
                "x": player_transform.location.x,
                "y": player_transform.location.y,
                "z": player_transform.location.z,
            })
        state_dict["speeding_details"] = []
        for time_stamp, player_transform in state.speeding_details:
            state_dict["speeding_details"].append({
                "time_stamp": time_stamp.elapsed_seconds,
                "x": player_transform.location.x,
                "y": player_transform.location.y,
                "z": player_transform.location.z,
            })
        #####################################
        state_dict["start_simulate_time"] = state.start_simulate_time
        state_dict["stop_simulate_time"] = state.stop_simulate_time
        #####################################

        if self.log_filename is None:
            filename = "{}_{}_{}_{}.json".format(state.campaign_cnt,
                                                 state.cycle_cnt, state.mutation, time.time())
        else:
            filename = self.log_filename

        if log_type == "queue":
            out_dir = self.conf.queue_dir

        # elif log_type == "error":
        # out_dir = self.conf.error_dir
        # elif log_type == "cov":
        # out_dir = self.conf.cov_dir

        with open(os.path.join(out_dir, filename), "w") as fp:
            json.dump(state_dict, fp)

        if self.conf.debug:
            print("[*] dumped")

        return filename

    def run_test(self, state):
        if self.conf.debug:
            print("[*] call executor.simulate()")
        if hasattr(state, "first_seed") and state.first_seed:
            print("[+] Info: First seed")
            seed_json = json.load(open(self.conf.seed_smy_dir + "/" + str(state.campaign_cnt) + ".json", "r"))
            self.ego_sp = carla.Transform(
                carla.Location(x=seed_json["sp_x"], y=seed_json["sp_y"], z=seed_json["sp_z"]),
                carla.Rotation(pitch=seed_json["pitch"], yaw=seed_json["yaw"], roll=seed_json["roll"])
            )
            self.ego_wp = carla.Transform(
                carla.Location(x=seed_json["wp_x"], y=seed_json["wp_y"], z=seed_json["wp_z"]),
                carla.Rotation(pitch=0, yaw=seed_json["wp_yaw"], roll=0)
            )
            self.weather = seed_json["wc"]
            self.puddles = seed_json["puddles"]
            for p in self.puddles:
                p["level"] = p["friction"]
                p["spawn_point"] = ((p["spawn_point"]["location"]["x"], p["spawn_point"]["location"]["y"],
                                     p["spawn_point"]["location"]["z"]), (0, 0, 0))
                p["size"] = (p["size"], p["size"], 1000)
            self.actors = []
            for w in seed_json["walkers"]:
                w["spawn_point"]["location"]["z"] += 3  # fix
                if w["behavior"] == "Autopilot":
                    nav_type = c.AUTOPILOT
                elif w["behavior"] == "Immobile":
                    nav_type = c.IMMOBILE
                elif w["behavior"] == "Linear":
                    nav_type = c.LINEAR
                self.actors.append({
                    "type": c.WALKER,
                    "nav_type": nav_type,
                    "spawn_point": ((w["spawn_point"]["location"]["x"], w["spawn_point"]["location"]["y"],
                                     w["spawn_point"]["location"]["z"]), (
                                        w["spawn_point"]["rotation"]["roll"], w["spawn_point"]["rotation"]["pitch"],
                                        w["spawn_point"]["rotation"]["yaw"])),
                    "dest_point": ((w["dest_point"]["location"]["x"], w["dest_point"]["location"]["y"],
                                    w["dest_point"]["location"]["z"]), (
                                       w["dest_point"]["rotation"]["roll"], w["dest_point"]["rotation"]["pitch"],
                                       w["dest_point"]["rotation"]["yaw"])) if nav_type == c.AUTOPILOT else None,
                    "speed": w["speed"],
                    "maneuvers": None,
                    "bp_id": w["bp"]["id"],
                    "color": w["color"],
                    "dp_time": w["dp_time"]
                })
            for v in seed_json["npc_vehicles"]:
                if v["behavior"] == "Autopilot":
                    nav_type = c.AUTOPILOT
                elif v["behavior"] == "Immobile":
                    nav_type = c.IMMOBILE
                elif v["behavior"] == "Linear":
                    nav_type = c.LINEAR
                self.actors.append({
                    "type": c.VEHICLE,
                    "nav_type": nav_type,
                    "spawn_point": ((v["spawn_point"]["location"]["x"], v["spawn_point"]["location"]["y"],
                                     v["spawn_point"]["location"]["z"]), (
                                        v["spawn_point"]["rotation"]["roll"], v["spawn_point"]["rotation"]["pitch"],
                                        v["spawn_point"]["rotation"]["yaw"])),
                    "dest_point": ((v["dest_point"]["location"]["x"], v["dest_point"]["location"]["y"],
                                    v["dest_point"]["location"]["z"]), (
                                       v["dest_point"]["rotation"]["roll"], v["dest_point"]["rotation"]["pitch"],
                                       v["dest_point"]["rotation"]["yaw"])) if nav_type == c.AUTOPILOT else None,
                    "speed": v["speed"],
                    "maneuvers": None,
                    "bp_id": v["bp"]["id"],
                    "color": v["color"],
                    "dp_time": v["dp_time"]
                })

        sp = self.ego_sp
        wp = self.ego_wp
        param = (
            self.conf,
            state,
            self.town,
            sp,
            wp,
            self.weather,
            self.puddles,
            self.actors
        )
        if os.path.exists("param.pick"):
            os.remove("param.pick")
        if os.path.exists("retval.pick"):
            os.remove("retval.pick")
        if os.path.exists("state.pick"):
            os.remove("state.pick")
        dill.dump(param, open("param.pick", "wb"))
        os.system("python executor.py")
        if not os.path.exists("retval.pick") or not os.path.exists("state.pick"):
            return -1, state
        ret = dill.load(open("retval.pick", "rb"))
        state = dill.load(open("state.pick", "rb"))
        exec_end_time = datetime.datetime.now()
        state.exec_end_time = exec_end_time.strftime('%Y-%m-%d %H:%M:%S')

        # print("after sim", time.time())

        # print("before logging", time.time())
        log_filename = self.dump_states(state, log_type="queue")
        self.log_filename = log_filename
        # print("after logging", time.time())

        if state.spawn_failed:
            obj = state.spawn_failed_object
            if isinstance(obj, int):
                if obj == 0:
                    print("failed object: ego car")
                else:
                    print("failed object:{}".format(obj))
            else:
                # don't try to spawn an infeasible actor in the next run
                # XXX: and we need a map of coordinates that represent
                #      spawn-feasibility
                print("failed object:{}".format(obj))
                if "level" in obj:
                    self.puddles.remove(obj)
                else:
                    self.actors.remove(obj)
            return -1, state

        # print("before error checking", time.time())
        if self.conf.debug:
            print("----- Check for errors -----")
        error = False
        if self.conf.check_dict["crash"] and state.crashed:
            error = True
        if self.conf.check_dict["stuck"] and state.stuck:
            if self.conf.debug:
                print("Vehicle stuck:", state.stuck_duration)
            error = True
        if self.conf.check_dict["lane"] and state.laneinvaded:
            error = True
        if self.conf.check_dict["red"] and state.red_violation:
            error = True
        if self.conf.check_dict["speed"] and state.speeding:
            if self.conf.debug:
                print("Speeding: {} km/h".format(state.speed[-1]))
            error = True
        if self.conf.check_dict["other"] and state.other_error:
            if state.other_error == "timeout":
                if self.conf.debug:
                    print("Simulation took too long")
            elif state.other_error == "goal":
                if self.conf.debug:
                    print("Goal is too far:", state.other_error_val, "m")
            error = True

        # print("before file ops", time.time())
        if self.conf.agent_type == c.BASIC or self.conf.agent_type == c.BEHAVIOR or self.conf.agent_type == c.LEADERBOARD or self.conf.agent_type == c.INTERFUSER or self.conf.agent_type == c.LMDRIVE or self.conf.agent_type == c.AUTOWARE_0903:
            if error:
                shutil.copyfile(
                    os.path.join(self.conf.queue_dir, log_filename),
                    os.path.join(self.conf.error_dir, log_filename)
                )
                shutil.move(
                    f"{self.conf.cache_dir}/front.mp4",
                    os.path.join(
                        self.conf.cam_dir,
                        log_filename.replace(".json", "-front.mp4")
                    )
                )

                shutil.move(
                    f"{self.conf.cache_dir}/top.mp4",
                    os.path.join(
                        self.conf.cam_dir,
                        log_filename.replace(".json", "-top.mp4")
                    )
                )
            if self.conf.cov_name is not None:
                if len(self.conf.cov_name) > 0:
                    cover_dir = [os.path.join(self.conf.cache_dir, f) for f in self.conf.cov_name]
                    for cover in cover_dir:
                        if os.path.exists(cover) and 'info' not in cover:
                            shutil.move(cover, os.path.join(self.conf.cov_dir, cover.split('/')[-1]))
                        else:
                            total_name = os.path.join(self.conf.cache_dir,
                                                      cover.split('/')[-1].split('_')[0] + '_total.info')
                            shutil.copy(cover, os.path.join(self.conf.cov_dir, cover.split('/')[-1]))
                            os.rename(cover, total_name)
            # cover_dir = glob.glob(os.path.join(self.conf.cache_dir, 'coverage_html'))[0]
            # shutil.copy(cover_dir, os.path.join(self.conf.cov_dir, "coverage_html"))

        # print("after file ops", time.time())

        if not self.conf.function.startswith("eval"):
            if ret == 128:
                return 128, state

        if error:
            self.found_error = True
            return 1, state

        if state.num_frames <= c.FRAME_RATE:
            # Trap for unlikely situation where test target didn't load
            # but we somehow got here.
            print("[-] Not enough data for scoring ({} frames)".format(
                state.num_frames))
            return -1, state

        # print("before scoring", time.time())
        if self.conf.debug:
            print("----- Scoring -----")
            # print("[debug] # frames:", state.num_frames)
            # print("[debug] elapsed time:", state.elapsed_time)
            # print("[debug] dist:", state.min_dist)
        np.set_printoptions(precision=3, suppress=True)

        # Attributes
        speed_list = np.array(state.speed)
        acc_list = np.diff(speed_list)

        Vx_list = np.array(state.lon_speed_list)
        Vy_list = np.array(state.lat_speed_list)
        SWA_list = np.array(state.steer_angle_list)

        # filter & process attributes
        Vx_light = get_vx_light(Vx_list)
        Ay_list = get_ay_list(Vy_list)
        Ay_diff_list = get_ay_diff_list(Ay_list)
        Ay_heavy = get_ay_heavy(Ay_list)
        SWA_diff_list = get_swa_diff_list(Vy_list)
        SWA_heavy_list = get_swa_heavy(SWA_list)
        Ay_gain = get_ay_gain(SWA_heavy_list, Ay_heavy)
        Ay_peak = get_ay_peak(Ay_gain)
        frac_drop = get_frac_drop(Ay_gain, Ay_peak)
        abs_yr = get_abs_yr(state.yaw_rate_list)

        deductions = 0

        # avoid infinitesimal md
        if int(state.min_dist) > 100 or int(state.min_dist) == 0:
            md = 0
        else:
            md = (1 / int(state.min_dist))

        ha = int(check_hard_acc(acc_list))
        hb = int(check_hard_braking(acc_list))
        ht = int(check_hard_turn(Vy_list, SWA_list))

        deductions += ha + hb + ht + md

        # check oversteer and understeer
        os_thres = 4
        us_thres = 4
        num_oversteer = 0
        num_understeer = 0
        for fid in range(len(Vy_list) - 2):
            SWA_diff = SWA_diff_list[fid]
            Ay_diff = Ay_diff_list[fid]
            yr = abs_yr[fid]

            Vx = Vx_light[fid]
            SWA2 = SWA_heavy_list[fid]
            fd = frac_drop[fid]
            os_level = get_oversteer_level(SWA_diff, Ay_diff, yr)
            us_level = get_understeer_level(fd)

            # TODO: add unstable event detection (section 3.5.1)

            if os_level >= os_thres:  # type: ignore
                if Vx > 5 and Ay_diff > 0.1:
                    num_oversteer += 1
                    # print("OS @%d %.2f (SWA %.4f Ay %.4f AVz %.4f Vx %.4f)" %(
                    # fid, os_level, SWA_diff, Ay_diff, yr, Vx))
            if us_level >= us_thres:  # type: ignore
                if Vx > 5 and SWA2 > 10:
                    num_understeer += 1
                    # print("US @%d %.2f (SA %.4f FD %.4f Vx %.4f)" %(
                    # fid, us_level, sa2, fd, Vx))

        ovs = int(num_oversteer)
        uds = int(num_understeer)
        deductions += ovs + uds
        state.deductions = {
            "ha": ha, "hb": hb, "ht": ht, "os": ovs, "us": uds, "md": md
        }

        self.driving_quality_score = -deductions

        print("[*] driving quality score: {}".format(
            self.driving_quality_score))

        # print("after scoring", time.time())

        with open(os.path.join(self.conf.score_dir, log_filename), "w") as fp:
            json.dump(state.deductions, fp)
        if ret == 666:
            return 666, state
        return ret, state


if __name__ == "__main__":
    # call run_test from this module only for debugging
    ts = TestScenario()
    ts.run_test()
