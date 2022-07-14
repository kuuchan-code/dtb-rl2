#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
import argparse

parser = argparse.ArgumentParser(description="訓練開始")

parser.add_argument("model", help="モデル")

args = parser.parse_args()

udid = "790908812299"
# device = "auto"
device = "cpu"

if args.model == "PPO":
    name_prefix = "_ppo_cnn_r4m11b"
    model_path = max(glob.glob("models/*ppo*.zip"), key=os.path.getctime)

    print(f"Load {model_path}")

    env = AnimalTower(udid=udid)

    model = PPO.load(
        path=model_path,
        env=env, tensorboard_log="tensorboard",
        device=device,
        print_system_info=True
    )

elif args.model == "A2C":

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
    save_freq=100, save_path="models",
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
