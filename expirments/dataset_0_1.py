import os
from scDML.scDML.metrics import evaluate_dataset
from expirments.load import load_to_adata_shaham_dataset
from main import ber_for_notebook

# import numpy as np
# import torch
# from sklearn.preprocessing import MinMaxScaler
# import scanpy as sc
# import torch.nn
# from metrics import batch_kl, compute_kbet, silhouette
# from plot import plot_data
# from unsupervised.utils import get_cdca_term

path_src = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_baseline.csv'
path_target = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_baseline.csv'
path_src_labels = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_baseline_label.csv'
path_target_labels = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_baseline_label.csv'

config = {
    "lr": 0.0001,  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": 0.2,  # or tune.choice([<list values>])
    "weight_decay": 0.2,  # or tune.choice([<list values>])
    "batch_size": 128,  # or tune.choice([<list values>])
    "epochs": 500,
    "save_weights": r"/weights/ber/dataset1/",
    "plots_dir": r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\plots\dataset10"}

if __name__ == "__main__":
    os.makedirs(config["save_weights"], exist_ok=True)
    os.makedirs(config["plots_dir"], exist_ok=True)

    adata = load_to_adata_shaham_dataset(path_src, path_target, path_src_labels, path_target_labels)

    evaluate_dataset(adata)    # Set the batch key for each cell
    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()

    source = adata1.X
    target = adata2.X
    source_labels = adata1.obs['celltype']
    target_labels = adata2.obs['celltype']

    adata_src_calibrated_target, adata_target_calibrated_src = ber_for_notebook(adata1, adata2,
                                                                                config)
