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
parser.add_argument(
    "-f", "--file", help="modelsディレクトリの中からファイル名指定 (一部でも可. 複数あれば最新のものを選ぶ)")
parser.add_argument("-s", "--udid", help="udid")
parser.add_argument("-d", "--device", help="device", default="auto")
parser.add_argument("--name", help="名前")
parser.add_argument("--env-verbose", help="詳細な出力", type=int, default=2)
parser.add_argument("--learning-rate", help="学習率", type=float)
parser.add_argument("--n-steps", help="n_steps", type=int)
parser.add_argument("--n-epochs", help="n_epochs", type=int)

args = parser.parse_args()

# udid = "790908812299" # 中古の黒
# udid = "482707805697"
# udid = "353010080451240" # 中古の灰色
# udid = "P3PDU18321001333" # あさひくんの
# device = "auto"
# device = "cpu"
x8_enabled = True

SRC_DIR = os.path.dirname(__file__)
MODELS_DIR = f"{SRC_DIR}/models/"

if args.model == "PPO":
    if args.file is None:
        model_path = max(
            glob.glob(f"{MODELS_DIR}/*ppo*"), key=os.path.getctime)
    else:
        model_path = max(
            glob.glob(f"{MODELS_DIR}/*{args.file}*"), key=os.path.getctime)
    # print(model_path)
    if args.name is None:
        mg = re.findall(f'models/(.+)_\d+_steps', model_path)
        name_prefix = f"_{mg[0]}"
    else:
        name_prefix = f"_ppo_cnn_r4m11b_{args.name}"

    # print(name_prefix)
    # exit()

    print(f"Load {model_path}")

    env = AnimalTower(udid=args.udid, log_prefix=name_prefix,
                      x8_enabled=x8_enabled, verbose=args.env_verbose)

    model = PPO.load(
        path=model_path,
        env=env,
        device=args.device,
    )
    if args.learning_rate is not None:
        model.learning_rate = args.learning_rate
    if args.n_steps is not None:
        model.n_steps = args.n_steps
    if args.n_epochs is not None:
        model.n_epochs = args.n_epochs

    # 一旦保存して再読込み
    # これでパラメータ変えられるか?
    model.save("models/_tmp")
    model = PPO.load(
        path="models/_tmp.zip",
        env=env, tensorboard_log="tensorboard",
        device=args.device,
        print_system_info=True
    )

    print(f"policy={model.policy}")
    print(f"learning_rate={model.learning_rate}")
    print(f"n_steps={model.n_steps}")
    print(f"batch_size={model.batch_size}")
    print(f"n_epochs={model.n_epochs}")
    print(f"gamma={model.gamma}")
    print(f"verbose={model.verbose}")
    print(f"device={model.device}")
    # exit()


elif args.model == "A2C":

    # 識別子
    # name_prefix = "_a2c_cnn_r4m11b"

    # 最新のモデルを読み込むように
    model_path = max(glob.glob("models/*a2c*.zip"), key=os.path.getctime)
    mg = re.findall(r'models/(.+)_\d+_steps', model_path)
    name_prefix = f"_{mg[0]}"
    # model_path = "models/a2c_cnn_r4m11b_54550_steps.zip"
    print(f"Load {model_path}")
    print(f"name_prefix={name_prefix}")

    env = AnimalTower(udid=args.udid,
                      log_prefix=name_prefix, x8_enabled=x8_enabled)

    model = A2C.load(
        path=model_path,
        env=env, tensorboard_log="tensorboard",
        device=args.device,
        print_system_info=True
    )
    # model.learning_rate = 0.0001

    print(f"policy={model.policy}")
    print(f"learning_rate={model.learning_rate}")
    print(f"gamma={model.gamma}")
    print(f"verbose={model.verbose}")
    print(f"device={model.device}")
    # exit()


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
finally:
    model.save("models/_end.zip")
