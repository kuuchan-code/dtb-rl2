#!/bin/bash

# pull rayproject/ray:latest-gpu
# sudo docker pull rayproject/ray:latest-gpu

# CPUを使う場合
# docker pull rayproject/ray:latest-cpu

# run rayproject/ray:latest-gpu
# sudo docker run -t -i --gpus all --shm-size=2.41gb -v ~/dtb-rl2:/home/ray/dtb-rl2 rayproject/ray:latest-gpu

# install tensorflow(in the docker)
# pip install tensorflow

# commit the container as rayproject/ray:latest-gpu-tensorflow
# sudo docker ps
# sudo docker commit コンテナID rayproject/ray:latest-gpu-tensorflow

# run rayproject/ray:latest-gpu-tensorflow
# sudo docker run -t -i --runtime=nvidia --shm-size=2.41gb -v ~/dtb-rl2:/home/ray/dtb-rl2 rayproject/ray:latest-gpu-tensorflow

# tensorflow with gpuとrllilbの動作確認(in the docker)
# python3 dtb-rl2/test/is_gpu_available.py
# python3 dtb-rl2/test/rllib.py

# OpenCVを動かすの
# sudo apt update
# sudo apt install libgl1-mesa-dev libopencv-dev

docker run --network host -it --runtime=nvidia --add-host=localhost_main:127.0.0.1 -v ~/dtb-rl2:/home/ray/dtb-rl2 rayproject/ray:latest-gpu-tensorflow