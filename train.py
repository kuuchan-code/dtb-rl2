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

name_prefix = "_a2c_cnn_rotate_move11_bin"
now_str = datetime.now().strftime("%Y%m%d%H%M%S")
print(f"{name_prefix}_{now_str}.csv")
# assert False
env = AnimalTower(log_path=f"log/{name_prefix}_{now_str}.csv")

# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
model = A2C.load(path=model_path,
                 env=env, tensorboard_log="tensorboard")
print(f"Loaded {model_path}")

# 新規作成
# model = A2C(policy='CnnPolicy', env=env,
#             verbose=1, tensorboard_log="tensorboard")

checkpoint_callback = CheckpointCallback(save_freq=50, save_path="models",
                                         name_prefix=name_prefix)
model.learn(total_timesteps=10000, callback=[checkpoint_callback])
