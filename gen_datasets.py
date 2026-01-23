import math
import os.path
import shutil
import random
import sys

import carla
import json
import copyreg
import dill


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


def lanemarkding_pickle(lanemarking):
    return carla.LaneMarking, (
        lanemarking.color,
        lanemarking.lane_change,
        lanemarking.type,
        lanemarking.width,
    )


def waypoint_pickle(waypoint):
    return carla.Waypoint, (
        waypoint.id,
        waypoint.transform,
        waypoint.road_id,
        waypoint.section_id,
        waypoint.lane_id,
        waypoint.s,
        waypoint.is_junction,
        waypoint.lane_width,
        waypoint.lane_change,
        waypoint.lane_type,
        waypoint.right_lane_marking,
        waypoint.left_lane_marking,
    )


copyreg.pickle(carla.Location, location_pickle)
copyreg.pickle(carla.Rotation, rotation_pickle)
copyreg.pickle(carla.Transform, transform_pickle)
copyreg.pickle(carla.Waypoint, waypoint_pickle)
copyreg.pickle(carla.LaneMarking, lanemarkding_pickle)


def create_sps_wps():
    if os.path.exists("towns_map"):
        shutil.rmtree("towns_map")
    os.mkdir("towns_map")
    # 连接到 CARLA 模拟器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    all_map = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    for map_name in all_map:
        client.load_world(map_name)
        world = client.load_world(map_name)
        carla_map = world.get_map()
        ##########################################################
        sps = carla_map.get_spawn_points()
        dill.dump(sps, open(f"towns_map/{map_name}.sps.pick", "wb"))
        ##########################################################
        sampling_distance = 5.0
        waypoints = carla_map.generate_waypoints(sampling_distance)
        road_waypoints = {}
        for waypoint in waypoints:
            if waypoint.is_junction:
                continue
            road_id = waypoint.road_id
            transform = waypoint.transform
            transform.location.z += 1.5
            vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
            try:
                ego = world.try_spawn_actor(vehicle_bp, transform)
                if ego is None:
                    print("Error in spawn")
                    continue
                ego.destroy()
            except Exception as e:
                print("Error in spawn")
                print(e)
                continue
            location = transform.location
            rotation = transform.rotation
            waypoint_info = {
                "location": {"x": location.x, "y": location.y, "z": location.z},
                "rotation": {
                    "pitch": rotation.pitch,
                    "yaw": rotation.yaw,
                    "roll": rotation.roll,
                },
                "lane_id": waypoint.lane_id,
                "lane_type": waypoint.lane_type,
            }
            if road_id not in road_waypoints:
                road_waypoints[road_id] = []
            road_waypoints[road_id].append(waypoint_info)
            print(f"This waypoint is ready for spawn: {waypoint_info}")
        road_waypoints_json = json.dumps(road_waypoints, indent=4)
        with open(f"towns_map/{map_name}.wps.json", "w") as json_file:
            json_file.write(road_waypoints_json)
        ##########################################################


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, carla.Location):
            return {"x": obj.x, "y": obj.y, "z": obj.z}
        elif isinstance(obj, carla.ActorBlueprint):
            return {"id": obj.id}
        elif isinstance(obj, carla.Waypoint):
            return {
                "location": {
                    "x": obj.transform.location.x,
                    "y": obj.transform.location.y,
                    "z": obj.transform.location.z,
                },
                "rotation": {
                    "pitch": obj.transform.rotation.pitch,
                    "yaw": obj.transform.rotation.yaw,
                    "roll": obj.transform.rotation.roll,
                },
                "lane_id": obj.lane_id,
                "lane_type": obj.lane_type,
            }
        elif isinstance(obj, carla.Transform):
            return {
                "location": {
                    "x": obj.location.x,
                    "y": obj.location.y,
                    "z": obj.location.z,
                },
                "rotation": {
                    "pitch": obj.rotation.pitch,
                    "yaw": obj.rotation.yaw,
                    "roll": obj.rotation.roll,
                },
            }
        else:
            return super().default(obj)


class Scenario:
    def __init__(self):
        self.map = None
        self.ego_bp = None
        self.sp_x = None
        self.sp_y = None
        self.sp_z = None
        self.pitch = None
        self.yaw = None
        self.roll = None
        self.wp_x = None
        self.wp_y = None
        self.wp_z = None
        self.wp_yaw = None
        self.wc = None
        self.walkers = None
        self.npc_vehicles = None
        self.puddles = None

    def init(self, map_name, wp1, wp2, ego_bp, wc, walkers, npc_vehicles, puddles):
        self.map = map_name
        self.ego_bp = ego_bp
        self.sp_x = wp1["location"]["x"]
        self.sp_y = wp1["location"]["y"]
        self.sp_z = wp1["location"]["z"]
        self.pitch = wp1["rotation"]["pitch"]
        self.yaw = wp1["rotation"]["yaw"]
        self.roll = wp1["rotation"]["roll"]
        self.wp_x = wp2["location"]["x"]
        self.wp_y = wp2["location"]["y"]
        self.wp_z = wp2["location"]["z"] - 1.5
        self.wp_yaw = wp2["rotation"]["yaw"]
        self.wc = wc
        self.walkers = walkers
        self.npc_vehicles = npc_vehicles
        self.puddles = puddles

    def to_json(self):
        return {
            "map": self.map,
            "ego_bp": self.ego_bp,
            "sp_x": self.sp_x,
            "sp_y": self.sp_y,
            "sp_z": self.sp_z,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "wp_x": self.wp_x,
            "wp_y": self.wp_y,
            "wp_z": self.wp_z,
            "wp_yaw": self.wp_yaw,
            "wc": self.wc,
            "walkers": self.walkers,
            "npc_vehicles": self.npc_vehicles,
            "puddles": self.puddles,
        }

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w") as json_file:
            json.dump(self.to_json(), json_file, indent=4, cls=CustomEncoder)

    def __eq__(self, other):
        if isinstance(other, Scenario):
            return self.to_json() == other.to_json()
        return False


def distance(wp1, wp2):
    if isinstance(wp1, carla.Location):
        wp1 = {"location": {"x": wp1.x, "y": wp1.y, "z": wp1.z}}
    if isinstance(wp1, carla.Waypoint):
        wp1 = {
            "location": {
                "x": wp1.transform.location.x,
                "y": wp1.transform.location.y,
                "z": wp1.transform.location.z,
            },
            "rotation": {
                "pitch": wp1.transform.rotation.pitch,
                "yaw": wp1.transform.rotation.yaw,
                "roll": wp1.transform.rotation.roll,
            },
        }
    if isinstance(wp1, carla.Transform):
        wp1 = {
            "location": {"x": wp1.location.x, "y": wp1.location.y, "z": wp1.location.z},
            "rotation": {
                "pitch": wp1.rotation.pitch,
                "yaw": wp1.rotation.yaw,
                "roll": wp1.rotation.roll,
            },
        }
    return math.sqrt(
        (wp1["location"]["x"] - wp2["location"]["x"]) ** 2
        + (wp1["location"]["y"] - wp2["location"]["y"]) ** 2
    )


def weather_conditions():
    return {
        "cloud": random.randint(0, 100),
        "rain": random.randint(0, 100),
        "wetness": random.randint(0, 100),
        "angle": random.randint(0, 360),
        "fog": random.randint(0, 100),
        "altitude": random.randint(-90, 90),
        "wind": random.randint(0, 100),
        "puddle": random.randint(0, 100),
    }


def walkers(world, ego_vehicle_loc, num=1):
    ret = []
    for idx in range(num):
        walker_bp = random.choice(
            world.get_blueprint_library().filter("walker.pedestrian.*")
        )
        sp_loc_strategy = random.choice(["Near", "Random"])
        sp_loc = random.choice(world.get_map().get_spawn_points())
        if sp_loc_strategy == "Near":
            try_times = 0
            while try_times < 100:
                if 5 < distance(sp_loc, ego_vehicle_loc) < 40:
                    break
                else:
                    sp_loc = random.choice(world.get_map().get_spawn_points())
                    try_times += 1
        behavior_type = random.choice(["Immobile", "Linear", "Autopilot"])
        if behavior_type == "Autopilot":
            dp_loc = random.choice(world.get_map().get_spawn_points())
        else:
            dp_loc = None
        if behavior_type == "Linear" or behavior_type == "Autopilot":
            speed = random.uniform(0, 10)
        else:
            speed = None
        if behavior_type == "Immobile":
            dp_time = random.randint(15, 30)
        else:
            dp_time = None
        ret.append(
            {
                "type": "Pedestrian",
                "behavior": behavior_type,
                "spawn_point": sp_loc,
                "dest_point": dp_loc,
                "speed": speed,
                "bp": walker_bp,
                "color": (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                "dp_time": dp_time,
            }
        )
    return ret


def npc_vehicles(world, ego_vehicle_loc, map_name, num=1):
    wps = json.load(open(f"towns_map/{map_name}.wps.json", "r"))
    ret = []
    for idx in range(num):
        vehicle_bp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
        sp_loc_strategy = random.choice(["Near", "Random"])
        sp_loc = random.choice(wps[random.choice(list(wps.keys()))])
        if sp_loc_strategy == "Near":
            while True:
                if 5 < distance(sp_loc, ego_vehicle_loc) < 40:
                    break
                else:
                    sp_loc = random.choice(wps[random.choice(list(wps.keys()))])
        behavior_type = random.choice(["Immobile", "Linear", "Autopilot"])
        if behavior_type == "Autopilot":
            dp_loc = random.choice(wps[random.choice(list(wps.keys()))])
        else:
            dp_loc = None
        if behavior_type == "Linear":
            speed = random.uniform(0, 30)
        else:
            speed = None
        if behavior_type == "Immobile":
            dp_time = random.randint(15, 30)
        else:
            dp_time = None
        ret.append(
            {
                "type": "NPC Vehicle",
                "behavior": behavior_type,
                "spawn_point": sp_loc,
                "dest_point": dp_loc,
                "speed": speed,
                "bp": vehicle_bp,
                "color": (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                "dp_time": dp_time,
            }
        )
    return ret


def puddles(ego_vehicle_loc, world, map_name, num=1):
    wps = json.load(open(f"towns_map/{map_name}.wps.json", "r"))
    ret = []
    for idx in range(num):
        sp_loc_strategy = random.choice(["Near", "Random"])
        sp_loc = random.choice(wps[random.choice(list(wps.keys()))])
        friction = random.randint(0, 200) / 100
        size = random.randint(0, 500)
        if sp_loc_strategy == "Near":
            while True:
                if 5 < distance(sp_loc, ego_vehicle_loc) < 40:
                    break
                else:
                    sp_loc = random.choice(wps[random.choice(list(wps.keys()))])
        ret.append(
            {
                "type": "Puddle",
                "spawn_point": sp_loc,
                "friction": friction,
                "size": size,
            }
        )
    return ret


def create_seed_scenarios(label="99"):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10)
    if os.path.exists(f"seed_scenarios{label}"):
        shutil.rmtree(f"seed_scenarios{label}")
    os.mkdir(f"seed_scenarios{label}")
    all_map = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    for map_name in all_map:
        world = client.load_world(map_name)
        os.mkdir(f"seed_scenarios{label}/{map_name}")
        wps = json.load(open(f"towns_map/{map_name}.wps.json", "r"))
        S = []
        while len(S) < 100:
            wp1_road_id, wp2_road_id = random.choices(list(wps.keys()), k=2)
            wp1 = random.choice(wps[wp1_road_id])
            wp2 = random.choice(wps[wp2_road_id])
            ego_bp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
            s = Scenario()
            s.init(
                map_name=map_name,
                wp1=wp1,
                wp2=wp2,
                wc=weather_conditions(),
                walkers=walkers(world=world, ego_vehicle_loc=wp1, num=1),
                npc_vehicles=npc_vehicles(
                    world=world, ego_vehicle_loc=wp1, map_name=map_name, num=1
                ),
                puddles=puddles(
                    ego_vehicle_loc=wp1, world=world, map_name=map_name, num=1
                ),
                ego_bp=ego_bp,
            )
            if 100 < distance(wp1, wp2) < 200 and s not in S:
                S.append(s)
        for idx, s in enumerate(S):
            s.to_json_file(
                os.path.join(f"seed_scenarios{label}", map_name, f"{idx}.json")
            )


def show_bp():
    # blueprint 'vehicle.dodge_charger.police' not found
    client = carla.Client("localhost", 5000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprints = [bp for bp in world.get_blueprint_library().filter("*")]
    for blueprint in blueprints:
        print(blueprint.id)


if __name__ == "__main__":
    create_sps_wps()
    create_seed_scenarios()
    show_bp()
