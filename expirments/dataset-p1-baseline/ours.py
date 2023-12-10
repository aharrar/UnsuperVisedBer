import os

import scanpy

# import numpy as np
# import torch
# from sklearn.preprocessing import MinMaxScaler
# import scanpy as sc
# from scDML.scDML.metrics import evaluate_dataset
# import torch.nn
from expirments.load import load_to_adata_shaham_dataset
# from main import ber_for_notebook
# from metrics import batch_kl, compute_kbet, silhouette
from plot import plot_data, get_pca_data, scatterHist
# from unsupervised.utils import get_cdca_term
print(f"here {os.curdir}")
plots_dir = r"../plots/ours/dataset-p1-baseline"

os.makedirs(plots_dir, exist_ok=True)
src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_baseline.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_baseline.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_baseline_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_baseline_label.csv'

#
# src_path = '../../data/Cytof/Person1Day1_baseline.csv'
# target_path = '../../data/Cytof/Person1Day2_baseline.csv'
# src_path_label = '../../data/Cytof/Person1Day1_baseline_label.csv'
# target_path_label = '../../data/Cytof/Person1Day2_baseline_label.csv'

config = {
    "lr": 0.01,  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": 0.2,  # or tune.choice([<list values>])
    "weight_decay": 0.2,  # or tune.choice([<list values>])
    "batch_size": 64,  # or tune.choice([<list values>])
    "epochs": 200,
    "save_weights": r"/weights/ber/dataset1/",
    "plots_dir": r"../plots/ours/dataset-p1-3m/"}

if __name__ == "__main__":
    os.makedirs(config["save_weights"], exist_ok=True)
    os.makedirs(config["plots_dir"], exist_ok=True)

    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)
    scanpy.pp.normalize_total(adata)

    # evaluate_dataset(adata)  # Set the batch key for each cell
    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()

    source = adata1.X
    target = adata2.X
    src_pca = get_pca_data(source)
    target_pca = get_pca_data(target)
    labels_b1 = adata1.obs['celltype']
    labels_b2 = adata2.obs['celltype']
    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="before-calibration-labels",
                name1='target', name2='src', plots_dir=plots_dir)
    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="before-calibrationp-batch",
                name1='target', name2='src', to_plot_labels=False, plots_dir=plots_dir)

    source_labels = adata1.obs['celltype']
    target_labels = adata2.obs['celltype']

    # adata_src_calibrated_target, adata_target_calibrated_src = ber_for_notebook(adata1, adata2,
    #                                                                             config)
    #
    # evaluate_dataset(adata_src_calibrated_target)  # Set the batch key for each cell
    # evaluate_dataset(adata_target_calibrated_src)  # Set the batch key for each cell
