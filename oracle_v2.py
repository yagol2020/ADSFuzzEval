import os
import time
import shutil

import carla
import sys
import json
import math
import numpy as np
import shapely
import loguru


def transform_2_dict(transform):
    return {
        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },
    }


def location_2_dict(location):
    return {
        "x": location.x,
        "y": location.y,
        "z": location.z,
    }


def lane_marking_2_dict(lane_marking):
    return {
        "color": (lane_marking.color),
        "lane_change": str(lane_marking.lane_change),
        "type": str(lane_marking.type),
        "width": lane_marking.width,
    }


def timestamp_2_dict(timestamp):
    return {
        "delta_seconds": timestamp.delta_seconds,
        "elapsed_seconds": timestamp.elapsed_seconds,
        "frame": timestamp.frame,
        "frame_count": timestamp.frame_count,
        "platform_timestamp": timestamp.platform_timestamp,
    }


def get_transform(actor):
    return actor.get_transform()


def calculate_velocity(actor, add_z=False):
    """
    Method to calculate the velocity of a actor
    """
    velocity_squared = actor.get_velocity().x ** 2
    velocity_squared += actor.get_velocity().y ** 2
    if add_z:
        velocity_squared += actor.get_velocity().z ** 2
    return math.sqrt(velocity_squared) * 3.6


def get_velocity(actor):
    return calculate_velocity(actor)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, carla.Transform):
            return transform_2_dict(obj)
        if isinstance(obj, carla.Timestamp):
            return timestamp_2_dict(obj)
        if isinstance(obj, carla.Location):
            return location_2_dict(obj)
        if isinstance(obj, carla.LaneMarking):
            return lane_marking_2_dict(obj)
        else:
            return super().default(obj)


class BaseOracle(object):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        self.ego_vehicle = ego_vehicle
        self.carla_world = carla_world
        self.start_sim_time = start_sim_time
        self.report = {
            "ego_vehicle_id": self.ego_vehicle.id,
            "ego_vehicle_type": self.ego_vehicle.type_id,
            "report": [],
            "report_v2": [],
        }
        self.oracle_name = None

    @loguru.logger.catch()
    def save_2_file(self, report_dir):
        path = os.path.join(report_dir, f"{self.oracle_name}.json")
        try:
            json.dump(
                self.report,
                open(path, "w", encoding="utf-8"),
                cls=CustomEncoder,
                indent=4,
            )
        except Exception as e:
            print("[-] Error:", e)
            sys.exit(-1)

    def update(self, time_stamp):
        pass

    def cleanup(self):
        pass


class CollisionOracle(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "collision"
        # 方法,carla自带的碰撞检测传感器
        self.carla_collision_sensor = None
        self.set_carla_collision_sensor()

    def set_carla_collision_sensor(self):
        carla_collision_sensor_bp = self.carla_world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.carla_collision_sensor = self.carla_world.spawn_actor(
            carla_collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.carla_collision_sensor.listen(
            lambda event: self.on_carla_collision_sensor(event)
        )

    def on_carla_collision_sensor(self, event):
        r = {
            "timestamp": event.timestamp,
            "collision_location": event.transform,
            "other_actor_id": event.other_actor.id,
            "other_actor_type": event.other_actor.type_id,
            "detected_time": time.time(),
            "start_sim_time": self.start_sim_time,
        }
        self._on_collision_v2(event)
        self.report["report"].append(r)

    def _on_collision_v2(self, event):
        detail_info = {}
        detail_info["frame"] = event.frame
        detail_info["timestamp"] = event.timestamp
        detail_info["transform"] = {
            "x": event.transform.location.x,
            "y": event.transform.location.y,
            "z": event.transform.location.z,
        }
        detail_info["start_sim_time"] = self.start_sim_time
        detail_info["detected_time"] = time.time()
        detail_info["other_actor"] = event.other_actor.type_id
        detail_info["is_ego"] = False
        normal_impulse = event.normal_impulse
        impulse_vector = carla.Vector3D(
            normal_impulse.x, normal_impulse.y, normal_impulse.z
        )

        vehicle_transform = event.transform
        forward_vector = vehicle_transform.get_forward_vector()
        right_vector = vehicle_transform.get_right_vector()

        impulse_magnitude = impulse_vector.length()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = (ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2) ** 0.5
        if impulse_magnitude > 0:

            normalized_impulse = impulse_vector / impulse_magnitude

            dot_product = (
                normalized_impulse.x * forward_vector.x
                + normalized_impulse.y * forward_vector.y
                + normalized_impulse.z * forward_vector.z
            )

            angle = math.acos(dot_product)

            angle_deg = math.degrees(angle)

            if angle_deg < 45:
                detail_info["part"] = "top"
                detail_info["is_ego"] = True
            elif angle_deg > 135:
                detail_info["part"] = "tail"
                detail_info["is_ego"] = False
            else:
                side_dot_product = (
                    normalized_impulse.x * right_vector.x
                    + normalized_impulse.y * right_vector.y
                    + normalized_impulse.z * right_vector.z
                )
                if side_dot_product > 0:
                    detail_info["part"] = "right"
                    if ego_speed > 0:
                        detail_info["is_ego"] = True
                    else:
                        detail_info["is_ego"] = False
                else:
                    detail_info["part"] = "left"
                    if ego_speed > 0:
                        detail_info["is_ego"] = True
                    else:
                        detail_info["is_ego"] = False
        self.report["report_v2"].append(detail_info)

    def cleanup(self):
        self.carla_collision_sensor.stop()
        self.carla_collision_sensor.destroy()


class LaneInvasionOracle(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "lane_invasion"
        self.carla_lane_invasion_sensor = None
        self.set_carla_lane_invasion_sensor()

    def set_carla_lane_invasion_sensor(self):
        carla_lane_invasion_sensor_bp = self.carla_world.get_blueprint_library().find(
            "sensor.other.lane_invasion"
        )
        self.carla_lane_invasion_sensor = self.carla_world.spawn_actor(
            carla_lane_invasion_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.carla_lane_invasion_sensor.listen(
            lambda event: self.on_carla_lane_invasion_sensor(event)
        )

    def on_carla_lane_invasion_sensor(self, event):
        r = {
            "timestamp": event.timestamp,
            "transform": transform_2_dict(event.transform),
            "invasion": event.crossed_lane_markings,
            "detected_time": time.time(),
            "start_sim_time": self.start_sim_time,
        }
        # self._on_invasion_v2(event)
        self.report["report"].append(r)

    def cleanup(self):
        self.carla_lane_invasion_sensor.stop()
        self.carla_lane_invasion_sensor.destroy()


class OutOfRoadOracle(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "out_of_road_oracle"
        # 方法, 来自leaderboard
        self._map = self.carla_world.get_map()
        self._offroad = False

        self._duration = 0  # default from leaderboard
        self._prev_time = None
        self._time_offroad = 0

    def update(self, time_stamp):
        current_location = self.ego_vehicle.get_location()

        # Get the waypoint at the current location to see if the actor is offroad
        drive_waypoint = self._map.get_waypoint(current_location, project_to_road=False)
        park_waypoint = self._map.get_waypoint(
            current_location, project_to_road=False, lane_type=carla.LaneType.Parking
        )
        if drive_waypoint or park_waypoint:
            self._offroad = False
        else:
            self._offroad = True

        # Counts the time offroad
        if self._offroad:
            loguru.logger.success("检测到Out of Road")
            r = {
                "timestamp": time_stamp,
                "offroad": True,
                "x": current_location.x,
                "y": current_location.y,
                "z": current_location.z,
            }
            return r
        return


class OnSidewalkOracle(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "on_sidewalk_oracle"
        self._map = self.carla_world.get_map()
        self._onsidewalk_active = False
        self._outside_lane_active = False

        self._actor_location = self.ego_vehicle.get_location()
        self._wrong_sidewalk_distance = 0
        self._wrong_outside_lane_distance = 0
        self._sidewalk_start_location = None
        self._outside_lane_start_location = None
        self._duration = 0
        self._prev_time = None
        self._time_outside_lanes = 0

    def update(self, time_stamp):

        # Some of the vehicle parameters
        current_tra = get_transform(self.ego_vehicle)
        current_loc = current_tra.location
        current_wp = self._map.get_waypoint(current_loc, lane_type=carla.LaneType.Any)

        # Case 1) Car center is at a sidewalk
        if current_wp.lane_type == carla.LaneType.Sidewalk:
            if not self._onsidewalk_active:
                self._onsidewalk_active = True
                self._sidewalk_start_location = current_loc

        # Case 2) Not inside allowed zones (Driving and Parking)
        elif (
            current_wp.lane_type != carla.LaneType.Driving
            and current_wp.lane_type != carla.LaneType.Parking
        ):

            # Get the vertices of the vehicle
            heading_vec = current_tra.get_forward_vector()
            heading_vec.z = 0
            heading_vec = heading_vec / math.sqrt(
                math.pow(heading_vec.x, 2) + math.pow(heading_vec.y, 2)
            )
            perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

            extent = self.ego_vehicle.bounding_box.extent
            x_boundary_vector = heading_vec * extent.x
            y_boundary_vector = perpendicular_vec * extent.y

            bbox = [
                current_loc + carla.Location(x_boundary_vector - y_boundary_vector),
                current_loc + carla.Location(x_boundary_vector + y_boundary_vector),
                current_loc
                + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
                current_loc
                + carla.Location(-1 * x_boundary_vector + y_boundary_vector),
            ]

            bbox_wp = [
                self._map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any),
            ]

            # Case 2.1) Not quite outside yet
            if (
                bbox_wp[0].lane_type
                == (carla.LaneType.Driving or carla.LaneType.Parking)
                or bbox_wp[1].lane_type
                == (carla.LaneType.Driving or carla.LaneType.Parking)
                or bbox_wp[2].lane_type
                == (carla.LaneType.Driving or carla.LaneType.Parking)
                or bbox_wp[3].lane_type
                == (carla.LaneType.Driving or carla.LaneType.Parking)
            ):

                self._onsidewalk_active = False
                self._outside_lane_active = False

            # Case 2.2) At the mini Shoulders between Driving and Sidewalk
            elif (
                bbox_wp[0].lane_type == carla.LaneType.Sidewalk
                or bbox_wp[1].lane_type == carla.LaneType.Sidewalk
                or bbox_wp[2].lane_type == carla.LaneType.Sidewalk
                or bbox_wp[3].lane_type == carla.LaneType.Sidewalk
            ):

                if not self._onsidewalk_active:
                    self._onsidewalk_active = True
                    self._sidewalk_start_location = current_loc

            else:
                distance_vehicle_wp = current_loc.distance(
                    current_wp.transform.location
                )

                # Case 2.3) Outside lane
                if distance_vehicle_wp >= current_wp.lane_width / 2:

                    if not self._outside_lane_active:
                        self._outside_lane_active = True
                        self._outside_lane_start_location = current_loc

                # Case 2.4) Very very edge case (but still inside driving lanes)
                else:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False

        # Case 3) Driving and Parking conditions
        else:
            # Check for false positives at junctions
            if current_wp.is_junction:
                distance_vehicle_wp = math.sqrt(
                    math.pow(current_wp.transform.location.x - current_loc.x, 2)
                    + math.pow(current_wp.transform.location.y - current_loc.y, 2)
                )

                if distance_vehicle_wp <= current_wp.lane_width / 2:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False
                # Else, do nothing, the waypoint is too far to consider it a correct position
            else:

                self._onsidewalk_active = False
                self._outside_lane_active = False

        # Update the distances
        distance_vector = self.ego_vehicle.get_location() - self._actor_location
        distance = math.sqrt(
            math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2)
        )

        if distance >= 0.02:  # Used to avoid micro-changes adding to considerable sums
            self._actor_location = self.ego_vehicle.get_location()

            if self._onsidewalk_active:
                self._wrong_sidewalk_distance += distance
            elif self._outside_lane_active:
                # Only add if car is outside the lane but ISN'T in a junction
                self._wrong_outside_lane_distance += distance

        # Register the sidewalk event
        if not self._onsidewalk_active and self._wrong_sidewalk_distance > 0:
            r = {
                "timestamp": time_stamp,
                "type": "on_sidewalk_infraction",
                "sidewalk_start_location": self._sidewalk_start_location,
                "wrong_sidewalk_distance": self._wrong_sidewalk_distance,
            }
            loguru.logger.success("检测到On Sidewalk")
            return r

        # Register the outside of a lane event
        if not self._outside_lane_active and self._wrong_outside_lane_distance > 0:
            r = {
                "timestamp": time_stamp,
                "type": "outside_lane_infraction",
                "outside_lane_start_location": self._outside_lane_start_location,
                "wrong_outside_lane_distance": self._wrong_outside_lane_distance,
            }
            loguru.logger.success("检测到On Sidewalk")
            return r

        return


class WrongLaneOracle(BaseOracle):
    MAX_ALLOWED_ANGLE = 120.0
    MAX_ALLOWED_WAYPOINT_ANGLE = 150.0

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "wrong_lane_oracle"
        #
        self._map = self.carla_world.get_map()
        self._last_lane_id = None
        self._last_road_id = None

        self._in_lane = True
        self._wrong_distance = 0
        self._actor_location = self.ego_vehicle.get_location()
        self._previous_lane_waypoint = self._map.get_waypoint(
            self.ego_vehicle.get_location()
        )
        self._wrong_lane_start_location = None

    def update(self, time_stamp):
        temp_r = []
        lane_waypoint = self._map.get_waypoint(self.ego_vehicle.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if (
            self._last_road_id != current_road_id
            or self._last_lane_id != current_lane_id
        ) and not lane_waypoint.is_junction:
            next_waypoint = lane_waypoint.next(2.0)[0]

            if not next_waypoint:
                return

            previous_lane_direction = (
                self._previous_lane_waypoint.transform.get_forward_vector()
            )
            current_lane_direction = lane_waypoint.transform.get_forward_vector()

            p_lane_vector = np.array(
                [previous_lane_direction.x, previous_lane_direction.y]
            )
            c_lane_vector = np.array(
                [current_lane_direction.x, current_lane_direction.y]
            )

            waypoint_angle = math.degrees(
                math.acos(
                    np.clip(
                        np.dot(p_lane_vector, c_lane_vector)
                        / (
                            np.linalg.norm(p_lane_vector)
                            * np.linalg.norm(c_lane_vector)
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )

            if waypoint_angle > self.MAX_ALLOWED_WAYPOINT_ANGLE and self._in_lane:

                self.test_status = "FAILURE"
                self._in_lane = False
                self._wrong_lane_start_location = self._actor_location

            else:
                # Reset variables
                self._in_lane = True

            # Continuity is broken after a junction so check vehicle-lane angle instead
            if self._previous_lane_waypoint.is_junction:

                vector_wp = np.array(
                    [
                        next_waypoint.transform.location.x
                        - lane_waypoint.transform.location.x,
                        next_waypoint.transform.location.y
                        - lane_waypoint.transform.location.y,
                    ]
                )

                vector_actor = np.array(
                    [
                        math.cos(
                            math.radians(self.ego_vehicle.get_transform().rotation.yaw)
                        ),
                        math.sin(
                            math.radians(self.ego_vehicle.get_transform().rotation.yaw)
                        ),
                    ]
                )

                vehicle_lane_angle = math.degrees(
                    math.acos(
                        np.clip(
                            np.dot(vector_actor, vector_wp)
                            / (np.linalg.norm(vector_wp)),
                            -1.0,
                            1.0,
                        )
                    )
                )

                if vehicle_lane_angle > self.MAX_ALLOWED_ANGLE:
                    self.test_status = "FAILURE"
                    self._in_lane = False
                    self._wrong_lane_start_location = self.ego_vehicle.get_location()

        # Keep adding "meters" to the counter
        distance_vector = self.ego_vehicle.get_location() - self._actor_location
        distance = math.sqrt(
            math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2)
        )

        if (
            distance >= 0.02
        ):  # Used to avoid micro-changes adding add to considerable sums
            self._actor_location = self.ego_vehicle.get_location()

            if not self._in_lane and not lane_waypoint.is_junction:
                self._wrong_distance += distance

        # Register the event
        if self._in_lane and self._wrong_distance > 0:
            r = {
                "timestamp": time_stamp,
                "type": "wrong_lane_infraction",
                "wrong_lane_start_location": self._wrong_lane_start_location,
                "wrong_distance": self._wrong_distance,
                "current_lane_id": current_lane_id,
                "current_road_id": current_road_id,
            }
            loguru.logger.success("检测到Wrong Lane")
            temp_r.append(r)
            self._wrong_distance = 0

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._previous_lane_waypoint = lane_waypoint
        return temp_r


class RunRedLightOracle(BaseOracle):
    DISTANCE_LIGHT = 15  # fork from leaderboard

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "run_red_light_clb"
        # 方法, 借鉴leaderboard的Criterion
        self._map = self.carla_world.get_map()
        self._list_traffic_lights = []
        self._last_red_light_id = None
        all_actors = self.carla_world.get_actors()
        for _actor in all_actors:
            if "traffic_light" in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString(
            [(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)]
        )
        line2 = shapely.geometry.LineString(
            [(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)]
        )
        inter = line1.intersection(line2)

        return not inter.is_empty

    def update(self, time_stamp):
        transform = get_transform(self.ego_vehicle)
        location = transform.location
        if location is None:
            loguru.logger.debug("在闯红灯检测中, 获取车辆位置失败")
            return

        veh_extent = self.ego_vehicle.bounding_box.extent.x

        tail_close_pt = self.rotate_point(
            carla.Vector3D(-0.8 * veh_extent, 0.0, location.z), transform.rotation.yaw
        )
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(
            carla.Vector3D(-veh_extent - 1, 0.0, location.z), transform.rotation.yaw
        )
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:

            center_loc = carla.Location(center)

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:

                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = get_transform(self.ego_vehicle).get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = (
                    ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
                )

                # Check the lane until all the "tail" has passed
                if (
                    tail_wp.road_id == wp.road_id
                    and tail_wp.lane_id == wp.lane_id
                    and dot_ve_wp > 0
                ):
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(
                        carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z),
                        yaw_wp + 90,
                    )
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(
                        carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z),
                        yaw_wp - 90,
                    )
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line(
                        (tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)
                    ):
                        message = (
                            f"The ego vehicle {self.ego_vehicle} is running a red traffic light({traffic_light}). "
                            f"The timestamp is {time_stamp}. The time detected is {time.time() - self.start_sim_time}"
                        )
                        location = traffic_light.get_transform().location
                        report = {
                            "tf_id": traffic_light.id,
                            "x": location.x,
                            "y": location.y,
                            "z": location.z,
                            "timestamp": time_stamp,
                            "detected_time": time.time(),
                            "message": message,
                            "start_sim_time": self.start_sim_time,
                        }
                        self.report["report"].append(report)
                        self._last_red_light_id = traffic_light.id
                        break


class RunRedLightOracle2(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "run_red_light_fuzz"
        self.on_red = False
        self.on_red_speed = list()
        self.last_red_light = None
        self.loc = list()

    def update(self, time_stamp):
        if self.ego_vehicle.is_at_traffic_light():
            traffic_light = self.ego_vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # print(f"[Red Light(Fuzz)]: The ego vehicle is running a red traffic light[{traffic_light}]")
                if self.on_red:
                    # print("[Red Light(Fuzz)]: Record the running red light speed")
                    self.on_red_speed.append(
                        calculate_velocity(self.ego_vehicle, add_z=True)
                    )
                    self.loc.append(self.ego_vehicle.get_location())
                else:
                    # print("[Red Light(Fuzz)]: First time into running red light, ready to record")
                    self.on_red = True
                    self.on_red_speed = list()
                    self.loc = list()
                    self.last_red_light = traffic_light
        else:
            if self.on_red:
                # print(
                #     "[Red Light(Fuzz)]: The ego vehicle is not running a red traffic light, stop recording, start analysis")
                self.on_red = False
                stopped_at_red = False
                for i, ors in enumerate(self.on_red_speed):
                    if ors < 0.1:
                        stopped_at_red = True
                        # print(
                        #     f"[Red Light(Fuzz)]: The ego vehicle stopped at {self.last_red_light} at {self.loc[i]}, current speed is {ors}, it is safe")
                if not stopped_at_red:
                    message = (
                        f"The ego vehicle {self.ego_vehicle} is running a red traffic light({self.last_red_light}). "
                        f"The timestamp is {time_stamp}. The time detected is {time.time() - self.start_sim_time}"
                    )
                    r = {
                        "timestamp": time_stamp,
                        "stopped_at_red": True,
                        "detected_time": time.time(),
                        "message": message,
                        "x": self.loc[0].x,
                        "y": self.loc[0].y,
                        "z": self.loc[0].z,
                        "speed": self.on_red_speed,
                        "tf_id": self.last_red_light.id,
                        "start_sim_time": self.start_sim_time,
                    }
                    self.report["report"].append(r)
                    self.last_red_light = None


def point_inside_boundingbox(point, bb_center, bb_extent):
    # pylint: disable=invalid-name
    A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
    B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
    D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
    M = carla.Vector2D(point.x, point.y)

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = AM.x * AB.x + AM.y * AB.y
    ab_ab = AB.x * AB.x + AB.y * AB.y
    am_ad = AM.x * AD.x + AM.y * AD.y
    ad_ad = AD.x * AD.x + AD.y * AD.y

    return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad


class RunStopSignOracle(BaseOracle):
    # fork from leaderboard code
    PROXIMITY_THRESHOLD = 50.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "run_stop_sign"
        # 方法, 借鉴leaderboard的Criterion
        self._map = self.carla_world.get_map()
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False
        self._affected_by_stop = False
        all_actors = self.carla_world.get_actors()
        for _actor in all_actors:
            if "traffic.stop" in _actor.type_id:
                self._list_stop_signs.append(_actor)

    def is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self.WAYPOINT_STEP)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None

        ve_tra = get_transform(self.ego_vehicle)
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_actor_affected_by_stop(self.ego_vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def update(self, time_stamp):
        """
        Check if the actor is running a red light
        """
        temp_r = []
        location = self.ego_vehicle.get_location()
        if location is None:
            loguru.logger.debug(f"在检测stop sign时，无法获取actor的location")
            return

        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self._stop_completed:
                # did the ego-vehicle stop?
                current_speed = get_velocity(self.ego_vehicle)
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_location = self._target_stop_sign.get_location()
                stop_extent = self._target_stop_sign.trigger_volume.extent

                if point_inside_boundingbox(location, stop_location, stop_extent):
                    self._affected_by_stop = True

            if not self.is_actor_affected_by_stop(
                self.ego_vehicle, self._target_stop_sign
            ):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed and self._affected_by_stop:
                    # did we stop?
                    stop_location = self._target_stop_sign.get_transform().location
                    r = {
                        "timestamp": time_stamp,
                        "id": self._target_stop_sign.id,
                        "x": stop_location.x,
                        "y": stop_location.y,
                        "z": stop_location.z,
                    }
                    loguru.logger.success("检测到Run Stop Sign")
                    temp_r.append(r)

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False
        return temp_r


class StuckOracle(BaseOracle):
    MIN_SPEED = 1  # km/h, fork from DriveFuzz
    STUCK_DURATION = 60 * 20  # second

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super(StuckOracle, self).__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "stuck"
        # 方法，思路来自DriveFuzz
        self.stuck_duration = 0

    def update(self, time_stamp):
        vel = calculate_velocity(self.ego_vehicle, add_z=True)
        if vel < self.MIN_SPEED:
            self.stuck_duration += 1
        else:
            self.stuck_duration = 0
        if self.stuck_duration > self.STUCK_DURATION:
            r = {
                "timestamp": time_stamp,
                "stuck": True,
                "detected_time": time.time(),
                "start_sim_time": self.start_sim_time,
            }
            self.report["report"].append(r)


class StuckOracleV2(BaseOracle):
    MIN_SPEED = 1.0  # km/h
    STUCK_DURATION = 60 * 20  # tick-based（假设 20Hz）
    BLOCK_DISTANCE = 20.0  # meters
    BLOCK_DURATION = 60 * 20  # tick-based
    ANGLE_THRESHOLD = 90.0  # degrees

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super(StuckOracleV2, self).__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "stuck_v2"
        self.stuck_duration = 0
        self.other_stuck_durations = {}  # {actor_id: stuck_tick_count}

    def _calc_speed(self, vehicle):
        """Return speed in km/h"""
        vel = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed

    def _is_in_front_sector(self, ego_tf, ego_loc, other_loc):
        vec = other_loc - ego_loc
        vec_norm = math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)
        if vec_norm < 1e-6:
            return False
        # angle between ego forward and vector to other
        forward = ego_tf.get_forward_vector()
        dot = forward.x * vec.x + forward.y * vec.y + forward.z * vec.z
        angle = math.degrees(
            math.acos(
                dot / (vec_norm * math.sqrt(forward.x**2 + forward.y**2 + forward.z**2))
            )
        )
        return angle <= self.ANGLE_THRESHOLD

    def update(self, time_stamp):
        # ego speed
        ego_speed = self._calc_speed(self.ego_vehicle)
        if ego_speed < self.MIN_SPEED:
            self.stuck_duration += 1
        else:
            self.stuck_duration = 0

        # check other vehicles
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        actors = self.carla_world.get_actors().filter("vehicle.*")

        blocked_by_other = False
        for veh in actors:
            if veh.id == self.ego_vehicle.id:
                continue
            speed = self._calc_speed(veh)

            if speed < self.MIN_SPEED:
                # update stuck counter
                self.other_stuck_durations[veh.id] = (
                    self.other_stuck_durations.get(veh.id, 0) + 1
                )
            else:
                self.other_stuck_durations[veh.id] = 0

            # 是否满足条件：在前方扇形、20m 内、且对方卡住很久
            if (
                self.other_stuck_durations[veh.id] > self.BLOCK_DURATION
                and ego_loc.distance(veh.get_location()) <= self.BLOCK_DISTANCE
                and self._is_in_front_sector(ego_tf, ego_loc, veh.get_location())
            ):
                blocked_by_other = True

        # ego stuck detection
        if self.stuck_duration > self.STUCK_DURATION:
            if not blocked_by_other:
                r = {
                    "timestamp": time_stamp,
                    "stuck": True,
                    "detected_time": time.time(),
                    "start_sim_time": self.start_sim_time,
                }
                self.report["report"].append(r)
                print("[StuckOracle] Ego is stuck (self issue).")
            else:
                print(
                    "[StuckOracle] Ego stopped, but blocked by another stuck vehicle → ignore."
                )


class NotReachGoalOracle(BaseOracle):
    """
    fork from drivefuzz
    """

    MIN_DIST = 10

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super().__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "goal_failed"

    def update(self, time_stamp, goal_loc, waypoint_queue):
        if len(waypoint_queue) > 0:
            return
        if isinstance(goal_loc, carla.Location):
            dist_to_goal = self.ego_vehicle.get_location().distance(goal_loc)
        else:
            dist_to_goal = self.ego_vehicle.get_location().distance(goal_loc.location)
        speed = calculate_velocity(self.ego_vehicle, add_z=True)
        if dist_to_goal >= self.MIN_DIST:
            message = (
                f"The ego vehicle {self.ego_vehicle} can not reach the goal, the distance to the goal is {dist_to_goal}, the speed is {speed} km/h. "
                f"The ego vehicle's waypoint is empty. The timestamp is {time_stamp}. The time detected is {time.time() - self.start_sim_time}"
            )
            r = {
                "timestamp": time_stamp,
                "not_reach_goal": True,
                "distance": dist_to_goal,
                "speed": speed,
                "detected_time": time.time(),
                "message": message,
                "start_sim_time": self.start_sim_time,
            }
            self.report["report"].append(r)


class SpeedingOracle(BaseOracle):
    # fork from drivefuzz, tm-fuzzer
    T = 3
    FRAME_RATE = 20

    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super(SpeedingOracle, self).__init__(ego_vehicle, carla_world, start_sim_time)
        self.oracle_name = "speeding"
        self.speed_limit = [0]
        self.frame_speed_lim_changed = 0

    def update(self, time_stamp):
        """
        NOTE: time_stamp = snapshot.frame
        """
        speed_limit = self.ego_vehicle.get_speed_limit()
        speed = calculate_velocity(self.ego_vehicle, add_z=True)
        if speed_limit != self.speed_limit[-1]:
            self.frame_speed_lim_changed = time_stamp
        self.speed_limit.append(speed_limit)

        if speed > speed_limit and time_stamp > (
            self.frame_speed_lim_changed + self.T * self.FRAME_RATE
        ):
            message = (
                f"The ego vehicle {self.ego_vehicle}'s speed is {speed} km/h, but the speed limit in this road is {speed_limit} km/h. "
                f"The ego vehicle speeding over {self.T * self.FRAME_RATE} frames, which is {self.T} seconds. "
                f"The timestamp is {time_stamp}. The time detected is {time.time() - self.start_sim_time}"
            )
            r = {
                "timestamp": time_stamp,
                "speeding": True,
                "detected_time": time.time(),
                "speed": speed,
                "speed_limit": speed_limit,
                "message": message,
                "start_sim_time": self.start_sim_time,
            }
            self.report["report"].append(r)


class LaneInvasionTimedDetector(BaseOracle):
    def __init__(self, ego_vehicle, carla_world, start_sim_time):
        super(LaneInvasionTimedDetector, self).__init__(
            ego_vehicle, carla_world, start_sim_time
        )
        self.oracle_name = "lane_invasion_v2"
        self.map = self.carla_world.get_map()
        self.ego = ego_vehicle
        self.allowed = 5.0
        self.angle_thresh = 90

        # 状态
        self.invasion_active = False  # 已进入对向车道并在计时中
        self.invasion_start = None
        self.violation_reported = False

        self.last_event = None

        # 创建并绑定 lane invasion sensor
        bp = self.carla_world.get_blueprint_library().find("sensor.other.lane_invasion")
        transform = carla.Transform()  # attach at vehicle origin
        self.sensor = self.carla_world.spawn_actor(bp, transform, attach_to=self.ego)
        self.sensor.listen(self._on_invasion)

    def _vec_dot(self, v1, v2):
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    def _vec_length(self, v):
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _angle_between_vectors_deg(self, v1, v2):
        # 安全处理数值误差
        dot = self._vec_dot(v1, v2)
        norm = self._vec_length(v1) * self._vec_length(v2)
        if norm == 0:
            return 0.0
        cosv = max(-1.0, min(1.0, dot / norm))
        return math.degrees(math.acos(cosv))

    def _is_waypoint_opposite_direction(self, veh_transform, waypoint):
        """
        比较车辆朝向与 waypoint 的前向向量：夹角大于 threshold -> 认为是对向车道
        """
        if waypoint is None:
            return False
        wp_forward = waypoint.transform.get_forward_vector()
        veh_forward = veh_transform.get_forward_vector()
        angle = self._angle_between_vectors_deg(veh_forward, wp_forward)
        return angle > self.angle_thresh

    def _on_invasion(self, event):
        """
        LaneInvasion sensor 回调：只在越线时触发一次（或每次跨线）
        在首次触发时判断是否进入对向车道，若是则开始计时。
        """
        self.last_event = event
        try:
            veh_tf = self.ego.get_transform()
            wp = self.map.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return

        # 已在计时中则不重复启动
        if self.invasion_active:
            return

        if wp is None:
            return

        if self._is_waypoint_opposite_direction(veh_tf, wp):
            # 进入对向车道，开始计时
            self.invasion_active = True
            self.invasion_start = time.time()
            self.violation_reported = False
            print("[LaneInvasionTimedDetector] Entered opposite lane — starting timer.")
        else:
            # 非对向（比如变道到同向邻道）— 不计时，仅记录越线但不启动 violation timer。
            print(
                "[LaneInvasionTimedDetector] Lane crossing but NOT opposite direction — ignore."
            )

    def update(self, timestamp):
        """
        每个仿真帧调用（在主 loop 中），用于检查是否仍在对向车道或是否超过 allowed 时间
        """
        if not self.invasion_active:
            return

        # get current waypoint
        try:
            veh_tf = self.ego.get_transform()
            wp = self.map.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            wp = None

        if wp is None:
            # 无法定位到路面（飞出地图等），按“仍在对向”继续计时；你可根据需要改为立即清除状态
            still_opposite = True
        else:
            still_opposite = self._is_waypoint_opposite_direction(veh_tf, wp)

        if still_opposite:
            elapsed = time.time() - self.invasion_start
            if elapsed > self.allowed and not self.violation_reported:
                self.violation_reported = True
                print(
                    f"[LaneInvasionTimedDetector] VIOLATION: stayed in opposite lane for {elapsed:.2f}s (> {self.allowed}s)"
                )
                r = {
                    "timestamp": self.last_event.timestamp,
                    "transform": transform_2_dict(self.last_event.transform),
                    "invasion": self.last_event.crossed_lane_markings,
                    "detected_time": time.time(),
                    "start_sim_time": self.start_sim_time,
                    "elapsed_time": elapsed,
                }
                self.report["report_v2"].append(r)
        else:
            # 已回到同向车道，视为安全结束（如果想记录时长可以在此处获取 elapsed）
            elapsed = time.time() - self.invasion_start
            print(
                f"[LaneInvasionTimedDetector] Returned to same-direction lane after {elapsed:.2f}s (<= {self.allowed}s). Safe."
            )
            # 清理状态
            self.invasion_active = False
            self.invasion_start = None
            self.violation_reported = False

    def destroy(self):
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except Exception:
            pass


class AllOracle:
    def __init__(self, ego_vehicle, carla_world, report_dir):
        self.start_sim_time = time.time()
        self.oracles = [
            CollisionOracle(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            SpeedingOracle(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            LaneInvasionOracle(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            LaneInvasionTimedDetector(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            RunRedLightOracle(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            RunRedLightOracle2(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            NotReachGoalOracle(
                ego_vehicle, carla_world, start_sim_time=self.start_sim_time
            ),
            StuckOracle(ego_vehicle, carla_world, start_sim_time=self.start_sim_time),
            StuckOracleV2(ego_vehicle, carla_world, start_sim_time=self.start_sim_time),
        ]
        self.ego_vehicle = ego_vehicle
        self.ego_traj = []
        self.report_dir = os.path.join(report_dir, "oracles", str(self.start_sim_time))
        if os.path.exists(self.report_dir):
            shutil.rmtree(self.report_dir)
        os.makedirs(self.report_dir)

    def update(self, snapshot, goal_loc, waypoint_queue):
        for oracle in self.oracles:
            if isinstance(oracle, SpeedingOracle):
                oracle.update(snapshot.frame)
            elif isinstance(oracle, NotReachGoalOracle):
                oracle.update(
                    snapshot.timestamp.elapsed_seconds,
                    goal_loc=goal_loc,
                    waypoint_queue=waypoint_queue,
                )
            else:
                oracle.update(snapshot.timestamp.elapsed_seconds)
        self.ego_traj.append(
            {
                "timestamp": snapshot.timestamp.elapsed_seconds,
                "x": self.ego_vehicle.get_location().x,
                "y": self.ego_vehicle.get_location().y,
                "z": self.ego_vehicle.get_location().z,
            }
        )

    def save_2_file(self):
        print("Copying param.pick files...")
        shutil.copy("param.pick", os.path.join(self.report_dir, "param.pick"))
        print("Copying param.pick files [done]...")
        print("Saving oracle reports...")
        for oracle in self.oracles:
            oracle.cleanup()
            print(f"{oracle} cleanup [done]...")
            oracle.save_2_file(self.report_dir)
            print(f"{oracle} save_2_file [done]...")
        print("Saving oracle reports [done]...")
        json.dump(
            self.ego_traj,
            open(os.path.join(self.report_dir, "ego_traj.json"), "w"),
            indent=4,
        )
