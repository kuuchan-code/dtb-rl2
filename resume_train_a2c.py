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

# 識別子
name_prefix = "_a2c_cnn_r4m11b"

# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
# model_path = "models/a2c_cnn_r4m11b_54550_steps.zip"
print(f"Load {model_path}")

env = AnimalTower(udid="482707805697",
                  log_prefix=name_prefix, x8_enabled=False)

# device = "cpu"
device = "auto"

model = A2C.load(
    path=model_path,
    env=env, tensorboard_log="tensorboard",
    device=device,
    print_system_info=True
)


checkpoint_callback = CheckpointCallback(
    save_freq=50, save_path="models",
    name_prefix=name_prefix
)
try:
    model.learn(total_timesteps=20000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
    raise e
except KeyboardInterrupt as e:
    print("キーボード割り込み")
    raise e
