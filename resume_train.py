#!/usr/bin/env python3
from ray.rllib.agents.dqn import ApexTrainer
from env import AnimalTower

from ray.tune.registry import register_env



def env_creator(env_config):
    env = AnimalTower()
    return env  # return an env instance

register_env("my_env", env_creator)

trainer = ApexTrainer(env="my_env", config={
    "framework": "tf",
    # R2D2 settings.
    "num_workers": 3,
    "target_network_update_freq": 500,
    "disable_env_checking": True,
    "timesteps_per_iteration": 25,
    "compress_observations": True,
    "learning_starts": 50,
    "train_batch_size": 64,
    "exploration_config": {"epsilon_timesteps": 10}
})

trainer.restore("~/ray_results/ApexTrainer_my_env_2022-07-12_01-57-00qk8uhjak/checkpoint_000013/")

for i in range(100):
    print(trainer.train())
    if i % 4 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)