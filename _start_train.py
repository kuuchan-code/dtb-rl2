#!/usr/bin/env python3
"""
初期から訓練開始
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
from datetime import datetime


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
