#!/usr/bin/env python3
"""
モデルの性能比較用
"""
import random as rd
from env import AnimalTower

# 中央の6
env = AnimalTower(udid="790908812299", log_prefix="all_6_560", x8_enabled=True)

obs = env.reset()
# 1000エピソードサンプルを集めたい
for i in range(1000):
    # action = 0
    obs, reward, done, info = env.step(27)
    env.render()
    if done:
        obs = env.reset()
