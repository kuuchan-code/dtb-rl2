#!/usr/bin/env python3
"""
なんかgym環境をチェックしてくれるやつ
ふつうにステップが走る、環境がおかしかったら文字列が出てくる
"""
# import ray
# from ray.rllib.utils import *
from env import AnimalTower, AnimalTowerDummy

# ray.rllib.utils.check_env(AnimalTower())


env = AnimalTowerDummy(debug=True)

# env.reset()

for i in range(10):
    env.reset()
    done = False
    while not done:
        obs, reward, done, _ = env.step(10)
        print(obs.shape, reward, done)
