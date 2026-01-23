import subprocess
import threading
from transforms3d.euler import euler2mat, quat2euler, euler2quat
import math
import time
import os

AUTOWARE_CONTAINER_NAME = "autoware"  # autoware
BRIDGE_CONTAINER_NAME = "autoware_bridge"  # autoware_bridge
ENV_SCRIPT_PATH = "env.sh"
ROS_SCRIPT_PATH = "/ros_entrypoint.sh"

autoware_start_ready = False
goal_ready = 0  # 0: not set, 1: good set, 2: error set
route_finished = False


def _stream_output(stream, cmd_name):
    global autoware_start_ready, goal_ready, route_finished
    for line in iter(stream.readline, b""):
        output_str = f"  [background] > {line.decode('utf-8').strip()}"
        if (
            "[system.system_monitor.net_monitor]: Failed to connect socket"
            in output_str
        ):
            continue
        if "[system.system_monitor.hdd_monitor]: socket connect error" in output_str:
            continue
        if (
            "[system.system_monitor.hdd_monitor]: Failed to unmount device : overlay"
            in output_str
        ):
            continue
        if (
            "[planning.scenario_planning.scenario_selector]: Waiting for route"
            in output_str
        ):
            autoware_start_ready = True
        if "Goal is not valid" in output_str:
            goal_ready = 2
    stream.close()


def _execute_command(
    command: str,
    container_name: str,
    background_run: bool,
    cmd_name="Def",
    easy_env=False,
    is_printer=True,
):
    global autoware_start_ready, goal_ready, route_finished
    if easy_env:
        full_command_inside_container = f"cd autoware_carla_launch && source ./install/setup.bash && source {ENV_SCRIPT_PATH} && {command}"
    else:
        full_command_inside_container = f"cd autoware_carla_launch && source ./install/setup.bash && source /ros_entrypoint.sh && source {ENV_SCRIPT_PATH} && {command}"

    command_to_run_on_host = [
        "docker",
        "exec",
        container_name,
        "bash",
        "-ic",
        full_command_inside_container,
    ]
    if is_printer:
        print(f"Running CMD: {command_to_run_on_host}")

    if background_run:
        try:
            process = subprocess.Popen(
                command_to_run_on_host,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            thread = threading.Thread(
                target=_stream_output, args=(process.stdout, cmd_name)
            )
            thread.daemon = True
            thread.start()
            print(f"✅ Process started in background with PID: {process.pid}")
            return process
        except FileNotFoundError:
            print("❌ Error: 'docker' command not found.")
            return False
        except Exception as e:
            print(f"❌ Error starting background process: {e}")
            return False
    else:
        try:
            result = subprocess.run(
                command_to_run_on_host, check=True, capture_output=True, text=True
            )
            if is_printer:
                print("✅ Command executed successfully!")
                print("--- Container Output ---")
                print(result.stdout.strip())
                if result.stderr:
                    print("--- Container Stderr ---")
                    print(result.stderr.strip())
            if cmd_name == "SetAuto" and "success=False" in result.stdout.strip():
                print("Set Autopilot model fail!")
                return False  # set autopilot mode fail
            if cmd_name == "Route" and "state: 4" in result.stdout.strip():
                goal_ready = 1
            if cmd_name == "Route" and "state: 6" in result.stdout.strip():
                route_finished = True
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing command: {e}")
            print("--- Error Details (stdout) ---")
            print(e.stdout)
            print("--- Error Details (stderr) ---")
            print(e.stderr)
            return False
        except FileNotFoundError:
            print("❌ Error: 'docker' command not found.")
            return False


def start_bridge():
    start_bridge_cmd = "pkill -f zenoh_carla_bridge && sleep 1 && ./script/bridge_ros2dds/run-bridge-only.sh"
    _execute_command(start_bridge_cmd, BRIDGE_CONTAINER_NAME, True)


def start_autoware():
    start_autoware_cmd = "pkill -f ros2 && pkill -f zenoh-bridge-ros2dds && sleep 1 && ./script/autoware_ros2dds/run-autoware.sh"
    _execute_command(start_autoware_cmd, AUTOWARE_CONTAINER_NAME, True, easy_env=True)


def stop_bridge():
    cmd = "pkill -f zenoh_carla_bridge"
    _execute_command(cmd, BRIDGE_CONTAINER_NAME, False)


def stop_autoware():
    cmd = "pkill -f ros2"
    _execute_command(cmd, AUTOWARE_CONTAINER_NAME, False)


def carla_transform_2_pose(tf):
    p_x, p_y, p_z = None, None, None
    ox, oy, oz, ow = None, None, None, None
    #####
    p_x, p_y, p_z = tf.location.x, -tf.location.y, tf.location.z
    carla_rotation = tf.rotation
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)
    quat = euler2quat(roll, pitch, yaw)
    ow, ox, oy, oz = quat[0], quat[1], quat[2], quat[3]
    return p_x, p_y, p_z, ox, oy, oz, ow


def send_goal_pose(dp):
    print("\n----- Sending Goal Pose -----")
    print(f"Transform gp {dp}")
    p_x, p_y, p_z, ox, oy, oz, ow = carla_transform_2_pose(tf=dp)
    message_payload = f"""
{{
    header: {{
        stamp: {{sec: 0, nanosec: 0}},
        frame_id: 'map'
    }},
    pose: {{
        position: {{x: {p_x}, y: {p_y}, z: {p_z}}},
        orientation: {{x: {ox}, y: {oy}, z: {oz}, w: {ow}}}
    }}
}}
"""

    ros_command = (
        f"ros2 topic pub --once "
        f"/planning/mission_planning/goal "
        f"geometry_msgs/msg/PoseStamped "
        f"'{message_payload.strip()}'"
    )

    _execute_command(ros_command, AUTOWARE_CONTAINER_NAME, False)


def change_to_autonomous_mode():
    print("\n----- Changing to Autonomous Mode -----")
    ros_command = (
        f"ros2 service call "
        f"/api/operation_mode/change_to_autonomous "
        f"autoware_adapi_v1_msgs/srv/ChangeOperationMode {{}}"
    )

    return _execute_command(
        ros_command, AUTOWARE_CONTAINER_NAME, False, cmd_name="SetAuto"
    )


monitor_routing_event = threading.Event()


def monitor_routing_state():
    cmd = "ros2 topic echo --once /planning/mission_planning/state"
    while not monitor_routing_event.is_set():
        _execute_command(
            cmd, AUTOWARE_CONTAINER_NAME, False, cmd_name="Route", is_printer=False
        )
        time.sleep(1)


def restart_ros_daemon():
    cmd = "ros2 daemon stop && ros2 daemon start"
    _execute_command(cmd, AUTOWARE_CONTAINER_NAME, False)


def zero_counters():
    cmd = "lcov --directory . --zerocounters"
    _execute_command(cmd, AUTOWARE_CONTAINER_NAME, False)


def lcov_once(label):
    cmd = f"lcov --directory . --capture --no-external --output-file lcov_coverage/{label}.info"
    _execute_command(cmd, AUTOWARE_CONTAINER_NAME, False,is_printer=False)
    if os.path.exists(f"/home/yy/autoware_carla_launch/lcov_coverage/{label}.info"):
        cmd = f"cd lcov_coverage && genhtml -o html_{label} {label}.info"
        _execute_command(cmd, AUTOWARE_CONTAINER_NAME, False,is_printer=False)
