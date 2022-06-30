#!/usr/bin/env python3
"""
なんかgym環境をチェックしてくれるやつ
ふつうにステップが走る、環境がおかしかったら文字列が出てくる
"""
from stable_baselines3.common.env_checker import check_env
from env import AnimalTower
check_env(env=AnimalTower())
