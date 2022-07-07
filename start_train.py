#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents.ddpg import ApexDDPGTrainer
from env import AnimalTower
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

trainer = ApexDDPGTrainer(env="my_env", config={
    "num_workers": 2,
    "learning_starts": 10,
    "framework": "tf",
    "clip_rewards": False,
    "exploration_config":{
        "ou_base_scale": 1.0
    },
    "n_step": 3,
    "target_network_update_freq": 100,
    "tau": 1.0,
    "evaluation_interval": None,
    "evaluation_duration": 10
})


# trainer.train()
# with open("/home/ray/dtb-rl2/a.txt", "w") as f:
#     print("あ", file=f)
for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
