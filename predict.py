#!/usr/bin/env python3
"""
モデルテスト用
"""
from stable_baselines3 import PPO, A2C
from env import AnimalTower, AnimalTowerDummy

# env = AnimalTower()
env = AnimalTowerDummy()

# pathを指定して任意の重みをロードする
model_path = "models/_a2c_cnn_r4m11b_color_10000_steps.zip"
print(f"Load {model_path}")

# env = AnimalTower(udid="482707805697",
#                   log_prefix=name_prefix, x8_enabled=False)
env = AnimalTowerDummy()
# model = PPO.load(path="ppo_logs/rotete_move_12_3200_steps", env=env)
model = A2C.load(path=model_path, env=env)

for i in range(50):
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
