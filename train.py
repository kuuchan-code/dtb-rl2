#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower

env = AnimalTower()
# 最新のモデルを読み込むように
#model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
#model = A2C.load(path=model_path,
#                 env=env, tensorboard_log="tensorboard")
model = A2C(policy='CnnPolicy', env=env,
            verbose=1, tensorboard_log="tensorboard")
#print(f"Loaded {model_path}")
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='models',
                                         name_prefix='_a2c_cnn_rotate_3d')
model.learn(total_timesteps=1500, callback=[checkpoint_callback])
