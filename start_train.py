#!/usr/bin/env python3
import gym
import ray
from ray.rllib.agents import dqn
from env import AnimalTower, udid_list
from selenium.common.exceptions import WebDriverException
from datetime import datetime
import json

from ray.tune.registry import register_env


def env_creator(env_config):
    env = AnimalTower()
    return env  # return an env instance


register_env("my_env", env_creator)

# for k, v in dqn.DEFAULT_CONFIG.copy().items():
#     print(f"{k}: {v}")
# assert False

trainer = dqn.R2D2Trainer(env="my_env", config={
    "framework": "tf",
    # R2D2 settings.
    "num_workers": 1,
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


for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
