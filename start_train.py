#!/usr/bin/env python3
<<<<<<< HEAD
import gym, ray
from ray.rllib.agents import dqn
=======
"""
初期から訓練開始
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
>>>>>>> sonoda-r4m11b
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
from datetime import datetime

<<<<<<< HEAD
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
        "learning_starts": 10,
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
=======

name_prefix = "_a2c_cnn_r4m3b_bin"
now_str = datetime.now().strftime("%Y%m%d%H%M%S")

env = AnimalTower(log_path=f"log/{name_prefix}_{now_str}.csv")

model = A2C(policy="CnnPolicy", env=env,
            verbose=2, tensorboard_log="tensorboard", learning_rate=0.0007)
checkpoint_callback = CheckpointCallback(
    save_freq=100, save_path="models",
    name_prefix=name_prefix
)

try:
    model.learn(total_timesteps=5000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
except KeyboardInterrupt as e:
    print("キーボード割り込み")
>>>>>>> sonoda-r4m11b
