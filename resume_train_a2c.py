#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower, AnimalTowerDummy
from selenium.common.exceptions import WebDriverException

# 識別子
# name_prefix = "_a2c_cnn_r4m11b"
name_prefix = "_a2c_dummy"

# 最新のモデルを読み込むように
model_path = max(glob.glob("models/*.zip"), key=os.path.getctime)
# model_path = "models/a2c_cnn_r4m11b_54550_steps.zip"
print(f"Load {model_path}")

# env = AnimalTower(udid="482707805697",
#                   log_prefix=name_prefix, x8_enabled=False)
env = AnimalTowerDummy()

save_freq = 50000
total_timesteps = 1000000

# device = "cpu"
device = "auto"

model = A2C.load(
    path=model_path,
    env=env, tensorboard_log="tensorboard",
    device=device,
    print_system_info=True
)


checkpoint_callback = CheckpointCallback(
    save_freq=save_freq, save_path="models",
    name_prefix=name_prefix
)
try:
    model.learn(total_timesteps=total_timesteps,
                callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
    raise e
except KeyboardInterrupt as e:
    print("キーボード割り込み")
    raise e
