import os

import pandas as pd

from plot import get_pca_data, scatterHist, plot2d
import pyliger
import scanpy as sc
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from expirments.load import load_and_pre_process_data

from metrics import eval_mmd

plots_dir = r"../plots/liger/dataset-p1-baseline"
os.makedirs(plots_dir, exist_ok=True)

# Load your data into two numpy arrays
src_path = '../../data/Cytof/Person1Day1_baseline.csv'
target_path = '../../data/Cytof/Person1Day2_baseline.csv'
src_path_label = '../../data/Cytof/Person1Day1_baseline_label.csv'
target_path_label = '../../data/Cytof/Person1Day2_baseline_label.csv'

if __name__ == "__main__":
    batch1, batch2 = load_and_pre_process_data(src_path, target_path)
    labels_b1 = np.loadtxt(src_path_label)
    labels_b2 = np.loadtxt(target_path_label)
    src_pca = get_pca_data(batch1)
    target_pca = get_pca_data(batch2)
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

    adata1 = sc.AnnData(X=batch1)
    adata2 = sc.AnnData(X=batch2)

    batch1_df = pd.read_csv(src_path, header=None)
    batch2_df = pd.read_csv(target_path, header=None)

    # Add headers for cells and genes
    batch1_df.index.name = 'Cell'  # Set cell index name
    batch2_df.index.name = 'Cell'  # Set cell index name

    # Create a Pyliger-compatible AnnData object
    adata1 = sc.AnnData((batch1_df.values))  # Transpose the DataFrame
    adata2 = sc.AnnData((batch2_df.values))  # Transpose the DataFrame
    adata1.obs.index.name = 'Cell'  # Set the cell index name
    adata1.var.index.name = 'Gene'  # Set the gene index name
    adata2.obs.index.name = 'Cell'  # Set the cell index name
    adata2.var.index.name = 'Gene'  # Set the gene index name
    adata1.uns['sample_name'] = 'Person1Day1'
    adata2.uns['sample_name'] = 'Person1Day2'
    sc.pp.normalize_total(adata1)  # Normalize data
    sc.pp.normalize_total(adata2)
    sc.pp.log1p(adata1)  # Log-transform data
    sc.pp.log1p(adata2)

    adata_list = [adata1, adata2]
    ifnb_liger = pyliger.create_liger(adata_list, remove_missing=False)

    pyliger.normalize(ifnb_liger)
    pyliger.select_genes(ifnb_liger)
    pyliger.scale_not_center(ifnb_liger)

    pyliger.optimize_ALS(ifnb_liger, k=14)
    pyliger.quantile_norm(ifnb_liger)
    pyliger.leiden_cluster(ifnb_liger, resolution=1)
    src_batch, target_batch = ifnb_liger.adata_list

    src_batch = np.array(src_batch.obsm['H'])
    target_batch = np.array(target_batch.obsm['H'])
    src_pca = get_pca_data(src_batch)
    target_pca = get_pca_data(target_batch)


    min_max_scaler_src = MinMaxScaler(feature_range=(-10, 10))
    min_max_scaler_target = MinMaxScaler(feature_range=(-10, 10))

    target_pca = min_max_scaler_src.fit_transform(target_pca)
    src_pca = min_max_scaler_target.fit_transform(src_pca)

    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", to_plot_labels=True, title="after-calibration-labels",
                name1='target', name2='src', plots_dir=plots_dir)

    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", to_plot_labels=False, title="after-calibration-batch",
                name1='target', name2='src', plots_dir=plots_dir)
