#!/usr/bin/env python3
import gym, ray
from ray.rllib.agents import dqn
from env import AnimalTower
from selenium.common.exceptions import WebDriverException
from datetime import datetime
import json

from ray.tune.registry import register_env


def env_creator(env_config):
    env = AnimalTower()
    return env  # return an env instance

register_env("my_env", env_creator)

# for k, v in dqn.DEFAULT_CONFIG.copy().items():
#     print(f"{k}: {v}")
# assert False

print(dqn.R2D2_DEFAULT_CONFIG)
trainer = dqn.R2D2Trainer(env="my_env", config={
    # "rollout_fragment_length": 4,
    "framework": "torch",
    # R2D2 settings.
    # "replay_buffer_config": {
        # "type": "MultiAgentReplayBuffer",
        # "storage_unit": "sequences",
        # "replay_burn_in": 20,  # 20
        # "zero_init_states": True
    # },
    #dueling: false
    # "lr": 0.0005,
    # Give some more time to explore.
    "exploration_config": {
        "epsilon_timesteps": 10  # 50000
    },
    "target_network_update_freq": 10,
    # "num_gpus": 1,
    # Wrap with an LSTM and use a very simple base-model.
    "model":{
        "fcnet_hiddens": [128],  # [64]
    #     "fcnet_activation": "linear",
        "use_lstm": True,
        "lstm_cell_size": 32,  # 64
        "max_seq_len": 2  # 20
    }
})


# trainer.train()
# with open("/home/ray/dtb-rl2/a.txt", "w") as f:
#     print("あ", file=f)
for i in range(10000):
    print(trainer.train())
    if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
