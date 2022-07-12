#!/usr/bin/env python3
from ray import tune
tune.run(
    "APEX",
    # other configuration
    # name="my_experiment",
    resume=True
)