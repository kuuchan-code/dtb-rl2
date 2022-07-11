#!/usr/bin/env python3
from ray import tune
tune.run(
    "R2D2",
    # other configuration
    # name="my_experiment",
    resume=True,
    local_dir="checkpoints"
)