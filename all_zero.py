#!/usr/bin/env python3
"""
モデルの性能比較用
"""
import random as rd
from env import AnimalTower

# 中央の6
env = AnimalTower(udid="790908812299", log_prefix="all_6_560", x8_enabled=True)

# 1000エピソードサンプルを集めたい
for i in range(1000-184):
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(27)
