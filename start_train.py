#!/usr/bin/env python3
"""
初期から訓練開始
"""
import glob
import os
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
import json
import argparse


parser = argparse.ArgumentParser(description="訓練開始")

parser.add_argument("model", help="モデル")
parser.add_argument("--name", help="名前")
parser.add_argument("-s", "--udid", help="udid")
parser.add_argument("-d", "--device", help="device", default="auto")
parser.add_argument("--env-verbose", help="詳細な出力", type=int, default=2)
parser.add_argument("--learning-rate", help="学習率", type=float, default=0.0003)
parser.add_argument("--n-steps", help="n_steps", type=int, default=2048)
parser.add_argument("--n-epochs", help="n_epochs", type=int, default=10)

args = parser.parse_args()

# udid = "482707805697"
# udid = "790908812299"
# udid = "353010080451240"
# device = "cpu"
# device = "auto"


if args.model == "DQN":
    name_prefix = "_dqn_cnn_r4m11b"

    # udidは適宜変更
    env = AnimalTower(udid="790908812299",
                      log_prefix=name_prefix, x8_enabled=True)

    # model = A2C(policy="CnnPolicy", env=env,
    #             verbose=2, tensorboard_log="tensorboard", learning_rate=0.0007)
    # 適当にパラメータセットしてみる
    # 学習開始のデフォルトが50000とかだったので, うまく学習できてなかった?
    model = DQN(
        policy="CnnPolicy", env=env, learning_rate=0.001, buffer_size=1000,
        learning_starts=500, batch_size=32, tau=0.5, gamma=0.999, train_freq=(10, "episode"),
        replay_buffer_class=None, optimize_memory_usage=True
    )
elif args.model == "PPO":
    learning_rate = args.learning_rate
    n_steps = args.n_steps
    n_epochs = args.n_epochs

    name_prefix = f"_ppo_cnn_r4m11b_lr{learning_rate}_ns{n_steps}_ne{n_epochs}"

    env = AnimalTower(udid=args.udid,
                      log_prefix=name_prefix, x8_enabled=True, verbose=args.env_verbose)

    model = PPO(policy="CnnPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps,
                batch_size=64, n_epochs=n_epochs, gamma=0.99, verbose=2, device=args.device)

    print(f"policy={model.policy}")
    print(f"learning_rate={model.learning_rate}")
    print(f"n_steps={model.n_steps}")
    print(f"batch_size={model.batch_size}")
    print(f"n_epochs={model.n_epochs}")
    print(f"gamma={model.gamma}")
    print(f"verbose={model.verbose}")
    print(f"device={model.device}")
else:
    exit(-1)

# 多分共通?
checkpoint_callback = CheckpointCallback(
    save_freq=500, save_path="models",
    name_prefix=name_prefix
)

try:
    model.learn(total_timesteps=30000, callback=[checkpoint_callback])
except WebDriverException as e:
    print("接続切れ?")
    raise e
except KeyboardInterrupt as e:
    print("キーボード割り込み")
    raise e
