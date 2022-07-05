#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import dqn
from env import AnimalTower


from ray.tune.registry import register_env

def env_creator(env_config):
    return AnimalTower()  # return an env instance

register_env("my_env", env_creator)

ray.init()
trainer = dqn.DQNTrainer(env="my_env", config={
    "env_config": {"n_step": 1, "noisy": True, "num_atoms": 2, "v_min": -10.0, "v_max": 10.0},
    "num_workers": 1,
})

while True:
    print(trainer.train())