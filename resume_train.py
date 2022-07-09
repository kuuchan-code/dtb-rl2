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
from selenium.common.exceptions import WebDriverException

# 識別子
name_prefix = "_a2c_cnn_r4m11b"
# 時刻
now_str = datetime.now().strftime("%Y%m%d%H%M%S")

# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
print(f"Load {model_path}")

# udidはメルカリで買った黒いやつ
env = AnimalTower(log_path=f"log/{name_prefix}_{now_str}.csv")

model = A2C.load(path=model_path,
                 env=env, tensorboard_log="tensorboard", device="cpu")


checkpoint_callback = CheckpointCallback(save_freq=50, save_path="models",
                                         name_prefix=name_prefix)
try:
    model.learn(total_timesteps=10000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
except KeyboardInterrupt as e:
    print("キーボード割り込み")
