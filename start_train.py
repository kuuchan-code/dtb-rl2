#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents.dqn import ApexTrainer
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

trainer = ApexTrainer(env="my_env", config={
    "framework": "tf",
    "target_network_update_freq": 100,
    "num_workers": 2,
    "learning_starts": 100
})


# trainer.train()
# with open("/home/ray/dtb-rl2/a.txt", "w") as f:
#     print("あ", file=f)
for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
