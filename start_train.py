#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower

name_prefix = "_a2c_cnn_rotate12_bin"
env = AnimalTower(log_path=name_prefix+".csv")
<<<<<<< HEAD:train.py
# 最新のモデルを読み込むように
#model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
#model = A2C.load(path=model_path,
#                env=env, tensorboard_log="tensorboard")
#print(f"Loaded {model_path}")
model = A2C(policy='CnnPolicy', env=env,
             verbose=1, tensorboard_log="tensorboard")
=======
model = A2C(policy='CnnPolicy', env=env,
            verbose=1, tensorboard_log="tensorboard")
>>>>>>> 2ac94929a6087d52aa48c95e73cac21fbf055ca4:start_train.py
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='models',
                                         name_prefix=name_prefix)
model.learn(total_timesteps=10000, callback=[checkpoint_callback])
