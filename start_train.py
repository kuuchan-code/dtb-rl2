#!/usr/bin/env python3

import ray.tune
from env import AnimalTower

ray.tune.run(
    "R2D2",
    config={
        "env":AnimalTower,
        "framework": "tf",
        # R2D2 settings.
        "num_workers": 1,
        "exploration_config": {"epsilon_timesteps": 40},
        "target_network_update_freq": 10,
        "model": {"use_lstm": True, "max_seq_len": 4},
        "disable_env_checking": True
    },
    checkpoint_freq=100,
    checkpoint_at_end=True,
    local_dir="checkpoints",
)
