#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException

name_prefix = "_a2c_cnn_r4m3_bin"
env = AnimalTower(log_path=name_prefix+".csv")

# 最新のモデルを読み込むように
# model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)

# おそらくr4m3
model_path = "models/model_rxmxb_10000_steps.zip"
model = A2C.load(path=model_path,
                 env=env, tensorboard_log="tensorboard")
print(f"Loaded {model_path}")

# model = A2C(policy='CnnPolicy', env=env,
#             verbose=1, tensorboard_log="tensorboard")
checkpoint_callback = CheckpointCallback(save_freq=50, save_path='models',
                                         name_prefix=name_prefix)
try:
    model.learn(total_timesteps=5000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
except KeyboardInterrupt as e:
    print("キーボード割り込み")
