#!/usr/bin/env python3
"""
モデルの性能比較用
"""
import random as rd
from env import AnimalTower

# ランダム
env = AnimalTower(log_episode_max=100)

obs = env.reset()
for i in range(10000):
    action = 0
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
