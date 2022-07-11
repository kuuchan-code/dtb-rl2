#!/usr/bin/env python3
"""
なんかgym環境をチェックしてくれるやつ
ふつうにステップが走る、環境がおかしかったら文字列が出てくる
"""
import ray
from ray.rllib.utils import *
from env import AnimalTower, AnimalTowerDummy

# ray.rllib.utils.check_env(AnimalTower())
env = AnimalTowerDummy()
env.reset()
