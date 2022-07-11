#!/usr/bin/env python3

import ray.tune
from env import AnimalTower

ray.tune.run(
    'R2D2',
    stop={
        'training_iteration': 10000,
    },
    config={
        'env':AnimalTower,
        'framework': 'tf',
        # R2D2 settings.
        'num_workers': 1,
        'compress_observations': True,
        'exploration_config': {'epsilon_timesteps': 40},
        'target_network_update_freq': 10,
        'model': {'use_lstm': True},
        'timesteps_per_iteration': 1,
        "disable_env_checking": True
    },
    checkpoint_freq=100,
    checkpoint_at_end=True,
    local_dir='checkpoints',
)
