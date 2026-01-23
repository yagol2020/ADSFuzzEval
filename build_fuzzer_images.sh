#!/bin/bash

echo "Building DriveFuzz image..."
cd DriveFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-InterFuser -t drivefuzz-interfuser . && cd ..
cd DriveFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-LMDrive -t drivefuzz-lmdrive . && cd ..

echo "Building TM-Fuzzer image..."
cd TM-Fuzzer && echo "switch to dir $PWD" && docker build -f Dockerfile-InterFuser -t tmfuzzer-interfuser . && cd ..
cd TM-Fuzzer && echo "switch to dir $PWD" && docker build -f Dockerfile-LMDrive -t tmfuzzer-lmdrive . && cd ..

echo "Building AV-Fuzzer image..."
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-AV-Fuzzer-InterFuser -t avfuzzer-interfuser . && cd ..
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-AV-Fuzzer-LMDrive -t avfuzzer-lmdrive . && cd ..

echo "Building SAMOTA image..."
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-SAMOTA-InterFuser -t samota-interfuser . && cd ..
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-SAMOTA-LMDrive -t samota-lmdrive . && cd ..

echo "Building ScenarioFuzz image..."
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-ScenarioFuzz-InterFuser -t scenariofuzz-interfuser . && cd ..
cd ScenarioFuzz && echo "switch to dir $PWD" && docker build -f Dockerfile-ScenarioFuzz-LMDrive -t scenariofuzz-lmdrive . && cd ..

echo "Done!"