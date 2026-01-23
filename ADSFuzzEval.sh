#!/bin/bash

cur_time=$(date +%s)

echo "Welcome ADSFuzzEval! Current Time is: $cur_time"

echo "The CPU Core Number is: $(nproc)"

fuzzer=$1
ads=$2
map_name=$3
map_label=$4
timeout=$5
carla_port_docker=$6
gpu_idx=$7
if [ -z "$fuzzer" ]; then
  echo "Please provide the fuzzer name as an argument."
  exit 1
fi

if [ -z "$ads" ]; then
  echo "Please provide the ADS name as an argument."
  exit 1
fi

if [ -z "$map_name" ]; then
  echo "Please provide the map name as an argument."
  exit 1
fi

if [ -z "$map_label" ]; then
  echo "Please provide the map label/index as an argument."
  exit 1
fi

if [ -z "$timeout" ]; then
  echo "Please provide the timeout as an argument."
  exit 1
fi

if [ -z "$carla_port_docker" ]; then
  echo "Please provide the port of carla as an argument."
  exit 1
fi

if [ -z "$gpu_idx" ]; then
  echo "Please provide the idx of gpu as an argument."
  exit 1
fi

echo "Fuzzer: $fuzzer"
echo "ADS: $ads"
echo "Map: $map_name"
echo "Map Label: $map_label"
echo "Timeout: $timeout"
echo "Carla Port: $carla_port_docker"
echo "GPU Index: $gpu_idx"

map_idx="${map_name//Town0/}" # number of Town, without the 'Town' prefix

if [ "$fuzzer" == "DriveFuzz" ]; then
  cmd="timeout "$timeout"h python -u fuzzer.py -s /seeds -o /output -t $ads"
  docker run --cpus="10" --rm --name "drivefuzz-runtime-$carla_port_docker" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER="$carla_port_docker" -e CUDA_VISIBLE_DEVICES="$gpu_idx" -v /var/run/docker.sock:/var/run/docker.sock -v ./seed_scenarios"$map_label"/"$map_name":/seeds -v ./results/DriveFuzz/"$map_name"/seed"$map_label"/"$ads"/"$timeout"h/"$cur_time":/output --privileged --net host --gpus all -e DISPLAY="$DISPLAY" drivefuzz-"$ads" -c "$cmd"
elif [ "$fuzzer" == "TM-Fuzzer" ]; then
  cmd="timeout "$timeout"h python -u fuzzer.py -s /seeds -o /output --town $map_idx"
  docker run --rm --name "tmfuzzer-runtime-$carla_port_docker" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER="$carla_port_docker" -e CUDA_VISIBLE_DEVICES="$gpu_idx" -v /var/run/docker.sock:/var/run/docker.sock -v ./seed_scenarios"$map_label"/"$map_name":/seeds -v ./results/TM-Fuzzer/"$map_name"/seed"$map_label"/"$ads"/"$timeout"h/"$cur_time":/output --privileged --net host --gpus all -e DISPLAY="$DISPLAY" tmfuzzer-"$ads" -c "$cmd"
elif [ "$fuzzer" == "AV-Fuzzer" ]; then
  cmd="timeout "$timeout"h python -u fuzzer_avfuzz.py -s /seeds -o /output --town $map_idx"
  docker run --rm --name "avfuzzer-runtime-$carla_port_docker" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER="$carla_port_docker" -e CUDA_VISIBLE_DEVICES="$gpu_idx" -v /var/run/docker.sock:/var/run/docker.sock -v ./seed_scenarios"$map_label"/"$map_name":/seeds -v ./results/AV-Fuzzer/"$map_name"/seed"$map_label"/"$ads"/"$timeout"h/"$cur_time":/output --privileged --net host --gpus all -e DISPLAY="$DISPLAY" avfuzzer-"$ads" -c "$cmd"
elif [ "$fuzzer" == "SAMOTA" ]; then
  cmd="timeout "$timeout"h python -u fuzzer_samota.py -s /seeds -o /output --town $map_idx"
  docker run --rm --name "samota-runtime-$carla_port_docker" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER="$carla_port_docker" -e CUDA_VISIBLE_DEVICES="$gpu_idx" -v /var/run/docker.sock:/var/run/docker.sock -v ./seed_scenarios"$map_label"/"$map_name":/seeds -v ./results/SAMOTA/"$map_name"/seed"$map_label"/"$ads"/"$timeout"h/"$cur_time":/output --privileged --net host --gpus all -e DISPLAY="$DISPLAY" samota-"$ads" -c "$cmd"
elif [ "$fuzzer" == "ScenarioFuzz" ]; then
  cmd="timeout "$timeout"h python -u fuzzer_eval.py --seed-smy-dir /seeds -o /output --town $map_idx"
  docker run --rm --name "scenariofuzz-runtime-$carla_port_docker" -e TZ="Asia/Shanghai" -e CARLA_PORT_DOCKER="$carla_port_docker" -e CUDA_VISIBLE_DEVICES="$gpu_idx" -v /var/run/docker.sock:/var/run/docker.sock -v ./seed_scenarios"$map_label"/"$map_name":/seeds -v ./results/ScenarioFuzz/"$map_name"/seed"$map_label"/"$ads"/"$timeout"h/"$cur_time":/output --privileged --net host --gpus all -e DISPLAY="$DISPLAY" scenariofuzz-"$ads" -c "$cmd"

else
  echo "Unknown fuzzer: $fuzzer"
  exit 1
fi

echo "ADSFuzzEval Done!"