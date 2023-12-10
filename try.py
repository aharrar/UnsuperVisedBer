import itertools
import random

import numpy as np
from numpy.setup import configuration

# Your original config
config = {
    "lr": np.random.uniform(1e-3, 1e-1, size=5).tolist(),
    "dropout": [0.2],
    "weight_decay": [0.2],
    "batch_size": [32, 64, 128],
    "epochs": [2],
    "save_weights": [r"../weights/ber/dataset1/"],
    "plots_dir": [r"../plots/ours/dataset-p1-3m/"]
}


def make_combinations_from_config(config):
    param_combinations = list(itertools.product(*config.values()))
    configurations = []
    for params in param_combinations:
        new_config = {key: value for key, value in zip(config.keys(), params)}
        configurations.append(new_config)

    return configurations

if __name__ == "__main__":
    ny = make_combinations_from_config(config)
    random_items = random.sample(ny, 10)
    print(random_items)
# print(param_combinations)
# # Create configurations for each combination
# Print the generated configurations
# for i, cfg in enumerate(configurations, start=1):
#     print(f"Configuration {i}: {cfg}")
