#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from datetime import datetime

# 識別子
name_prefix = "_a2c_cnn_r4m11b"
# 時刻
now_str = datetime.now().strftime("%Y%m%d%H%M%S")
print(f"{name_prefix}_{now_str}.csv")
# assert False
env = AnimalTower(log_path=f"log/{name_prefix}_{now_str}.csv")

# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
# model_path = "models/a2c_cnn_rotate_move11_bin_1950_steps.zip"

model = A2C.load(path=model_path,
                 env=env, tensorboard_log="tensorboard")
print(f"Loaded {model_path}")


checkpoint_callback = CheckpointCallback(save_freq=50, save_path="models",
                                         name_prefix=name_prefix)
try:
    model.learn(total_timesteps=10000, callback=[checkpoint_callback])
except KeyboardInterrupt as e:
    print("キーボード割り込み")
