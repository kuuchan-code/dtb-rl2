#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import dqn
from env import AnimalTower

from ray.tune.registry import register_env

def env_creator(env_config):
    return AnimalTower()  # return an env instance

register_env("my_env", env_creator)


trainer = dqn.DQNTrainer(env="my_env", config={
    "num_workers": 1,
    "num_atoms": 51,
    "noisy": True,
    "gamma": 0.99,
    "lr": .0001,
    "hiddens": [512],
    "rollout_fragment_length": 4,
    "train_batch_size": 32,
    "exploration_config": {
        "epsilon_timesteps": 2,
        "final_epsilon": 0.0,
    },
    "target_network_update_freq": 500,
    "replay_buffer_config":{
        "type": "MultiAgentPrioritizedReplayBuffer",
        "prioritized_replay_alpha": 0.5,
        "learning_starts": 500,
        "capacity": 50000
    },
    "n_step": 3,
    "model": {
        "grayscale": True,
        "zero_mean": False,
        "dim": 42
    }
})

for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)