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

trainer = dqn.DQNTrainer(env="my_env", config={
    "num_workers": 2,
    "num_atoms": 51,
    "noisy": True,
    "gamma": 0.99,
    "lr": .0001,
    "hiddens": [512],
    "rollout_fragment_length": 4,
    "train_batch_size": 1,
    "exploration_config": {
        "epsilon_timesteps": 2,
        "final_epsilon": 0.0,
    },
    "target_network_update_freq": 500,
    "replay_buffer_config":{
        "type": "MultiAgentPrioritizedReplayBuffer",
        "prioritized_replay_alpha": 0.5,
        "learning_starts": 10,
        "capacity": 50000
    },
    "n_step": 3,
    "model": {
        "grayscale": True,
        "zero_mean": False,
        "dim": 42
    },
    "disable_env_checking": True,
    "ignore_worker_failures": True
})


trainer.train()
with open("/home/ray/dtb-rl2/a.txt", "w") as f:
    print("あ", file=f)
# for i in range(10000):
#     print(trainer.train())
#     if i % 100 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
