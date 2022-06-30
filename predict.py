#!/usr/bin/env python3
"""
モデルテスト用
"""
from stable_baselines3 import PPO
from env import AnimalTower

env = AnimalTower()

# pathを指定して任意の重みをロードする
model = PPO.load(path="ppo_logs/rotete_move_12_3200_steps", env=env)

obs = env.reset()
for i in range(50):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
