#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import ppo
from env import AnimalTower


from ray.tune.registry import register_env

def env_creator(env_config):
    return AnimalTower()  # return an env instance

register_env("my_env", env_creator)

ray.init()
trainer = ppo.PPOTrainer(env="my_env", config={
    "env_config": {},  # config to pass to env class
    "num_workers": 1,
})

while True:
    print(trainer.train())