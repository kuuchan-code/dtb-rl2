#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import a3c
from env import AnimalTower


from ray.tune.registry import register_env

def env_creator(env_config):
    return AnimalTower()  # return an env instance

register_env("my_env", env_creator)

ray.init()
config = a3c.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 1
config["framework"] = "tf2"
config["log_level"] = "INFO"
trainer = a3c.A3CTrainer(env="my_env", config=config)

for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)