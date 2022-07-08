#!/usr/bin/env python3
"""
初期から訓練開始
"""
import glob
import os
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
from datetime import datetime
import json


name_prefix = "_dqn_cnn_r4m3b_bin"
now_str = datetime.now().strftime("%Y%m%d%H%M%S")

# udidは適宜変更
env = AnimalTower(
    # udid="482707805697",
    udid="CB512C5QDQ",
    log_path=f"log/{name_prefix}_{now_str}.csv",
    x8_enabled=True
)

# model = A2C(policy="CnnPolicy", env=env,
#             verbose=2, tensorboard_log="tensorboard", learning_rate=0.0007)
# 適当にパラメータセットしてみる
# 学習開始のデフォルトが50000とかだったので, うまく学習できてなかった?
model = DQN(
    policy="CnnPolicy", env=env, learning_rate=0.01, buffer_size=500,
    learning_starts=100, batch_size=64, tau=0.5, gamma=0.999, train_freq=(10, "episode")
)

checkpoint_callback = CheckpointCallback(
    save_freq=100, save_path="models",
    name_prefix=name_prefix
)

try:
    model.learn(total_timesteps=10000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
except KeyboardInterrupt as e:
    print("キーボード割り込み")
