#!/usr/bin/env python3
"""
モデルの性能比較用
"""
import random as rd
from env import AnimalTower

# r4m11bランダム
env = AnimalTower(udid="CB512C5QDQ",
                  log_prefix="random_r4m11b", x8_enabled=True)

obs = env.reset()
# 1000エピソードサンプルを集めたい
for i in range(1000):
    action = rd.randint(0, env.action_patterns-1)
    # action = 0
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
