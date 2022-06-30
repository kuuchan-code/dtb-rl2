#!/usr/bin/env python3
"""
訓練用
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower

env = AnimalTower()
model = PPO.load(path="ppo_logs/rotate__100_steps",
                 env=env, tensorboard_log="./ppo_tensorboard/")
# model = PPO(policy='CnnPolicy', env=env,
#             verbose=1, tensorboard_log="./ppo_tensorboard/")
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./ppo_zipfiles/',
                                         name_prefix='rotate')
model.learn(total_timesteps=1500, callback=[checkpoint_callback])
