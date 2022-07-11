#!/usr/bin/env python3
"""
なんかgym環境をチェックしてくれるやつ
ふつうにステップが走る、環境がおかしかったら文字列が出てくる
"""
import random
from time import sleep
import ray
from ray.rllib.utils import *
from env import AnimalTower, AnimalTowerDummy

ray.rllib.utils.check_env(AnimalTowerDummy())
env = AnimalTowerDummy()
# env.reset()
# env.step(10)

for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        obs, reward, done, _ = env.step(random.randint(0, 21))
        sleep(0.1)
