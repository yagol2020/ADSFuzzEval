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
        carla.Location(transform.location.x, transform.location.y, transform.location.z),
        carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
    )


def lanemarkding_pickle(lanemarking):
    return carla.LaneMarking, (
        lanemarking.color, lanemarking.lane_change, lanemarking.type, lanemarking.width
    )


def waypoint_pickle(waypoint):
    return carla.Waypoint, (
        waypoint.id, waypoint.transform, waypoint.road_id, waypoint.section_id, waypoint.lane_id, waypoint.s,
        waypoint.is_junction, waypoint.lane_width, waypoint.lane_change, waypoint.lane_type,
        waypoint.right_lane_marking, waypoint.left_lane_marking
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
    client = carla.Client('localhost', 2000)
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
                "location": {
                    "x": location.x,
                    "y": location.y,
                    "z": location.z
                },
                "rotation": {
                    "pitch": rotation.pitch,
                    "yaw": rotation.yaw,
                    "roll": rotation.roll
                },
                "lane_id": waypoint.lane_id,
                "lane_type": waypoint.lane_type
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
            return {
                "x": obj.x,
                "y": obj.y,
                "z": obj.z
            }
        elif isinstance(obj, carla.ActorBlueprint):
            return {
                "id": obj.id
            }
        elif isinstance(obj, carla.Waypoint):
            return {
                "location": {
                    "x": obj.transform.location.x,
                    "y": obj.transform.location.y,
                    "z": obj.transform.location.z
                },
                "rotation": {
                    "pitch": obj.transform.rotation.pitch,
                    "yaw": obj.transform.rotation.yaw,
                    "roll": obj.transform.rotation.roll
                },
                "lane_id": obj.lane_id,
                "lane_type": obj.lane_type
            }
        elif isinstance(obj, carla.Transform):
            return {
                "location": {
                    "x": obj.location.x,
                    "y": obj.location.y,
                    "z": obj.location.z
                },
                "rotation": {
                    "pitch": obj.rotation.pitch,
                    "yaw": obj.rotation.yaw,
                    "roll": obj.rotation.roll
                }
            }
        else:
            return super().default(obj)

if __name__ == "__main__":
    create_sps_wps()
