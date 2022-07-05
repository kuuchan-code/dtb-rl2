#!/bin/bash

# pull rayproject/ray:latest-gpu
sudo docker pull rayproject/ray:latest-gpu

# run rayproject/ray:latest-gpu
# sudo docker run -t -i --gpus all --shm-size=2.41gb -v ~/dtb-rl2:/home/ray/dtb-rl2 rayproject/ray:latest-gpu

# install tensorflow(in the docker)
# pip install tensorflow

# commit the container as rayproject/ray:latest-gpu-tensorflow
# sudo docker ps
# sudo docker commit コンテナID rayproject/ray:latest-gpu-tensorflow

# run rayproject/ray:latest-gpu-tensorflow
# sudo docker run -t -i --runtime=nvidia --shm-size=2.41gb -v ~/dtb-rl2:/home/ray/dtb-rl2 rayproject/ray:latest-gpu-tensorflow

# install tensorflow(in the docker)
# python3 dtb-rl2/test/rllib.py