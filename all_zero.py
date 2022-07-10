#!/usr/bin/env python3
"""
モデルの性能比較用
"""
import random as rd
from env import AnimalTower

# r4m11bランダム
env = AnimalTower(udid="CB512C5QDQ",
                  log_prefix="random_r4m11b", x8_enabled=True)

# 1000エピソードサンプルを集めたい
for i in range(1000-261):
    obs = env.reset()
    done = False
    while not done:
        action = rd.randint(0, env.action_patterns-1)
        obs, reward, done, info = env.step(action)
