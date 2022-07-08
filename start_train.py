#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import dqn
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

print(dqn.R2D2_DEFAULT_CONFIG)
trainer = dqn.R2D2Trainer(env="my_env", config={
    "framework": "tf",
    # R2D2 settings.
    "exploration_config": {
        "epsilon_timesteps": 30  # 10000
    },
    "target_network_update_freq": 9,  # 2500
    "model":{
        "fcnet_hiddens": [64, 64],  # [256, 256]
        "use_lstm": True,  # False
        "lstm_cell_size": 64,  # 256
        "max_seq_len": 2  # 20
    }
})


# trainer.train()
# with open("/home/ray/dtb-rl2/a.txt", "w") as f:
#     print("あ", file=f)
for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
