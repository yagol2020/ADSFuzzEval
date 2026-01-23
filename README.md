# ADSFuzzEval

## Acknowledgement

* [DriveFuzz](https://gitlab.com/s3lab-code/public/drivefuzz)
* [TM-Fuzzer](https://github.com/ldegao/TMfuzz)
* [AV-Fuzzer](https://github.com/cclinus/AV-Fuzzer)
* [ScenarioFuzz](https://github.com/AtongWang/ScenarioFuzz)
* [CARLA](https://github.com/carla-simulator/carla)
* [InterFuser](https://github.com/opendilab/InterFuser)
* [LMDrive](https://github.com/opendilab/LMDrive)
* [Autoware](https://github.com/autowarefoundation/autoware)
* [autoware_carla_launch](https://github.com/evshary/autoware_carla_launch)


### System Requirements

| Requirements | Version          |
|--------------|------------------|
| System OS    | Ubuntu 20.04 LTS |
| Docker       | 27.3.1           |
| CARLA        | 0.9.13           |

### Install Dependencies

1. Docker
2. Docker with Nvidia Support
3. Pull CARLA Docker Image
4. Download model weights for [InterFuser](https://github.com/opendilab/InterFuser) and [LMDrive](https://github.com/opendilab/LMDrive?tab=readme-ov-file).

### Create Fuzzer Images

```shell
./build_fuzzer_images.sh
```

### Build Autoware and Bridge Docker Image
We refer to the documentation of [autoware_carla_launch](https://autoware-carla-launch.readthedocs.io/en/latest/build.html)
```shell
git clone https://github.com/evshary/autoware_carla_launch.git
cd autoware_carla_launch
./container/run-bridge-docker.sh
# In the docker container
cd autoware_carla_launch
source env.sh
make prepare_bridge
make build_bridge

# open a new terminal and run the following command to start the bridge container
cd autoware_carla_launch
./container/run-autoware-docker.sh
# In the docker container
cd autoware_carla_launch
source env.sh
make prepare_autoware
make build_autoware
```

## Evaluation

We provide a script to facilitate the evaluation of different fuzzers. The script takes the following parameters:
* Fuzzer name: DriveFuzz, TMFuzzer, AVFuzzer, ScenarioFuzz
* ADS name: InterFuser, LMDrive, Autoware
* Town: Town01, Town02, Town03, Town04, Town05
* Dataset Index: 1, 2, ...
* Duration: e.g., 10h, 30m
* CARLA Port: e.g., 2000, 4000
* GPU Index: e.g., 0, 1

For example, to run DriveFuzz with InterFuser in Town01 for 10 hours using CARLA port 2000 and GPU index 1, use the following command:

```shell
sudo chmod 777 ADSFuzzEval.sh
./ADSFuzzEval.sh DriveFuzz InterFuser Town01 1 10h 2000 1
```

> Before testing on Autoware, you need to start the bridge and Autoware docker containers, as shown below:

```shell
# Start CARLA simulator
./run_carla_v2.sh -p 2000 -v 0.9.13
# Go inside bridge container
./container/run-bridge-docker.sh
# start the bridge
cd autoware_carla_launch
source env.sh
./script/bridge_ros2dds/run-bridge.sh
# Open a new terminal and go inside Autoware container
./container/run-autoware-docker.sh
# start Autoware
cd autoware_carla_launch
source env.sh
./script/autoware_ros2dds/run-autoware.sh
```

## Enable Improved Oracle and Detectors

We provide our improved detecots `oracle_v2.py`, which can be used to replace the original oracle in the fuzzer. To enable the improved oracle, you can use the following command:

```shell
# For example, replace the oracle in DriveFuzz
mv oracle_v2.py DriveFuzz/oracle.py
```

### FP Case in Collision with Vehicles

We provide a false positive case with InterFuser in Town01 in `fp_case_study/`. You can run the following command to reproduce the case:

```shell
# start a tmfuzzer-interfuser container
docker run --rm --name "tmfuzzer-runtime" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER=6000 -e CUDA_VISIBLE_DEVICES=1 -v /var/run/docker.sock:/var/run/docker.sock  --privileged --net host --gpus all tmfuzzer-interfuser
# copy the driving scenario file to the container
docker cp fp_case_study/param.pick tmfuzzer-runtime:/TM-Fuzzer/param.pick
# go inside the container and start simulation with the scenario file
docker exec -it tmfuzzer-runtime bash
cd TM-Fuzzer
python simulate.py
```

The front and top carmera views of the false positive case are saved in `/tmp/fuzzerdata-tmfuzz/`, you could copy them to your local machine using `docker cp` command, which is similar to the following videos:

| Front Camera View | Top Camera View |
|-------------------|-----------------|
| [front_camera_view.mp4](fp_case_study/front.mp4) | [top_camera_view.mp4](fp_case_study/top.mp4) |
| ![front](fp_case_study/front.gif) | ![top](fp_case_study/top.gif) |


## Known Issues

1. No protocol specified
    ```shell
    xhost +
    ./ADSFuzzEval.sh
    ```
2. Autoware is unstable / cannot locate the vehicle
   
A possible reason is that the Autoware container starts ros2 multiple times, but fails to clean up the environment after the simulation ends. You can try `pkill -f ros2` and then retry.
