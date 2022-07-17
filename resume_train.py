#!/usr/bin/env python3
"""
訓練用
"""
import glob
import os
import re
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
import argparse

parser = argparse.ArgumentParser(description="訓練開始")

parser.add_argument("model", help="モデル")
parser.add_argument("-s", "--udid", help="udid")

args = parser.parse_args()

# udid = "790908812299"
# udid = "482707805697"
device = "auto"
# device = "cpu"
x8_enabled = True

if args.model == "PPO":
    # name_prefix = "_ppo_cnn_r4m11b"
    model_path = max(glob.glob("models/*ppo*"), key=os.path.getctime)
    mg = re.findall(f'models/(.+)_\d+_steps', model_path)
    name_prefix = f"_{mg[0]}"

    # print(name_prefix)
    # exit()

    print(f"Load {model_path}")

    env = AnimalTower(udid=args.udid, log_prefix=name_prefix,
                      x8_enabled=x8_enabled)

    model = PPO.load(
        path=model_path,
        env=env, tensorboard_log="tensorboard",
        device=device,
        print_system_info=True
    )
    # 学習率変えてみる
    model.learning_rate = 0.0001

    print(f"policy={model.policy}")
    print(f"learning_rate={model.learning_rate}")
    print(f"n_steps={model.n_steps}")
    print(f"batch_size={model.batch_size}")
    print(f"n_epochs={model.n_epochs}")
    print(f"gamma={model.gamma}")
    print(f"verbose={model.verbose}")
    print(f"device={model.device}")


elif args.model == "A2C":

    # 識別子
    name_prefix = "_a2c_cnn_r4m11b"

    # 最新のモデルを読み込むように
    model_path = max(glob.glob("models/*a2c*.zip"), key=os.path.getctime)
    # model_path = "models/a2c_cnn_r4m11b_54550_steps.zip"
    print(f"Load {model_path}")

    env = AnimalTower(udid=args.udid,
                      log_prefix=name_prefix, x8_enabled=x8_enabled)

    model = A2C.load(
        path=model_path,
        env=env, tensorboard_log="tensorboard",
        device=device,
        print_system_info=True
    )


checkpoint_callback = CheckpointCallback(
    save_freq=200, save_path="models",
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
finally:
    model.save("models/_end.zip")
