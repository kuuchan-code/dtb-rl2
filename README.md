# dtb-rl2
Deep reinforcement learning on the small base of the Animal Tower.
## r4
r = [0, 4, 6, 8]
## conv_filterについて
https://docs.ray.io/en/ray-1.1.0/rllib-models.html

After preprocessing raw environment outputs, these preprocessed observations are then fed through a policy’s model. RLlib picks default models based on a simple heuristic: A vision network (TF or Torch) for observations that have a shape of length larger than 2 (for example, (84 x 84 x 3)), and a fully connected network (TF or Torch) for everything else. These models can be configured via the model config key, documented in the model catalog. Note that for the vision network case, you’ll probably have to configure conv_filters if your environment observations have custom sizes, e.g., "model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]} for 42x42 observations. Thereby, always make sure that the last Conv2D output has an output shape of [B, 1, 1, X] ([B, X, 1, 1] for Torch), where B=batch and X=last Conv2D layer’s number of filters, so that RLlib can flatten it. An informative error will be thrown if this is not the case.
