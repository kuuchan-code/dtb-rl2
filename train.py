#!/usr/bin/env python3
"""
訓練用
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower

env = AnimalTower()
model = PPO.load(path="ppo_logs/rotete_8_???_steps",
                 env=env, tensorboard_log="./ppo_dtb/")
# model = PPO(policy='CnnPolicy', env=env,
#             verbose=1, tensorboard_log="./ppo_tf/")
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./ppo_logs/',
                                         name_prefix='rotete_8')
model.learn(total_timesteps=1000, callback=[checkpoint_callback])
