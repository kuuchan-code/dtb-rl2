#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from ray.rllib.agents import dqn
from env import AnimalTower
from datetime import datetime
from selenium.common.exceptions import WebDriverException
import argparse

parser = argparse.ArgumentParser(description="訓練を再開")
parser.add_argument("file", help="読み込むcheckpointファイル")

args = parser.parse_args()

trainer = dqn.R2D2Trainer(env="my_env", config={
    "framework": "tf",
    # R2D2 settings.
    "num_workers": 3,
    "compress_observations": True,
    "exploration_config": {
        "epsilon_timesteps": 40  # 10000
    },
    "target_network_update_freq": 10,  # 2500
    "model": {
        "fcnet_hiddens": [256, 256],  # [256, 256]
        "use_lstm": True,  # False
        "lstm_cell_size": 256,  # 256
        "max_seq_len": 20  # 20
    },
    "disable_env_checking": True,
    "timesteps_per_iteration": 1
})

trainer.restore(args.file)

for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)