import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from main import ber_for_notebook
from expirments.load import load_to_adata_shaham_dataset, get_batch_from_adata

from scDML.scDML.metrics import evaluate_dataset
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

print(f"here {os.curdir}")
plots_dir = r"../plots/ours/dataset-p1-3m/"
os.makedirs(plots_dir, exist_ok=True)

# Load your data into two numpy arrays
src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month_label.csv'
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "lr": [0.001,0.005],  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [ 0.2],  # or tune.choice([<list values>])
    "batch_size": [ 64,128],  # or tune.choice([<list values>])
    "epochs": [200],
    "coef_1": [64,32,10,5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset-p1-3m-TR1Y/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset-p1-3m-TR1Y/")]
}
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=5)




if __name__ == "__main__":

    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)

    # evaluate_dataset(adata)  # Set the batch key for each cell

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata,plot_dir=config["plots_dir"], title='before-calibrationp')
        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))
        adata1, adata2 = get_batch_from_adata(adata)
        adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                    adata2=adata2)
        plot_adata(adata_target_calibrated_src,plot_dir=config["plots_dir"], title='after-calibration-target_calibrated_src')
        plot_adata(adata_src_calibrated_target,plot_dir=config["plots_dir"], title='after-calibration-src_calibrated_target')

        # evaluate_dataset(adata_src_calibrated_target).to_csv(
        #     os.path.join(config["plots_dir"], "adata_src_calibrated_target.csv"))  # Set the batch key for each cell
        # evaluate_dataset(adata_target_calibrated_src).to_csv(
        #     os.path.join(config["plots_dir"], "adata_target_calibrated_src.csv"))  # Set the batch key for each cell
