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
import json


name_prefix = "_dqn_cnn_r4m11b"

# udidは適宜変更
env = AnimalTower(udid="790908812299", log_prefix=name_prefix)

# model = A2C(policy="CnnPolicy", env=env,
#             verbose=2, tensorboard_log="tensorboard", learning_rate=0.0007)
# 適当にパラメータセットしてみる
# 学習開始のデフォルトが50000とかだったので, うまく学習できてなかった?
model = DQN(
    policy="CnnPolicy", env=env, learning_rate=0.001, buffer_size=1000,
    learning_starts=500, batch_size=32, tau=0.5, gamma=0.999, train_freq=(10, "episode")
)

checkpoint_callback = CheckpointCallback(
    save_freq=100, save_path="models",
    name_prefix=name_prefix
)

try:
    model.learn(total_timesteps=10000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
    raise e
except KeyboardInterrupt as e:
    print("キーボード割り込み")
    raise e
