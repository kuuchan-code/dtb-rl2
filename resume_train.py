#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower

name_prefix = "_r12m1sb_rewarddecay"
env = AnimalTower(log_path=name_prefix+".csv")
# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
model = A2C.load(path=model_path,
                env=env, tensorboard_log="tensorboard")
print(f"Loaded {model_path}")
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='models',
                                         name_prefix=name_prefix)
model.learn(total_timesteps=1500, callback=[checkpoint_callback])
