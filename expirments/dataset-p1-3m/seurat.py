# Import the necessary packages
import os

import scanpy as sc
import numpy as np

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize

from expirments.load import load_and_pre_process_data, load_to_adata_shaham_dataset, get_batch_from_adata
from expirments.utils import plot_adata
from metrics import eval_mmd
from plot import get_pca_data, scatterHist

# from scDML.scDML.metrics import evaluate_dataset

plots_dir = r"../plots/seurat/dataset-p1-3m"

os.makedirs(plots_dir, exist_ok=True)
path_src = '../../data/Cytof/Person1Day1_3month.csv'
path_target = '../../data/Cytof/Person1Day2_3month.csv'
src_path_label = '../../data/Cytof/Person1Day1_3month_label.csv'
target_path_label = '../../data/Cytof/Person1Day2_3month_label.csv'

if __name__ == "__main__":
    src_data_without_labels, target_data_without_labels = load_and_pre_process_data(path_src, path_target)

    labels_b1 = np.loadtxt(src_path_label)
    labels_b2 = np.loadtxt(target_path_label)
    adata = load_to_adata_shaham_dataset(path_src, path_target, src_path_label, target_path_label)
    scaler = MinMaxScaler()
    adata.X = scaler.fit_transform(adata.X)
    sc.pp.normalize_total(adata)
    adata1, adata2 = get_batch_from_adata(adata)
    print(eval_mmd(torch.tensor(adata1.X), torch.tensor(adata2.X)))

    plot_adata(adata)
    # Identify highly variable genes using the Seurat flavor
    sc.pp.highly_variable_genes(adata, batch_key='batch', flavor='seurat_v3', n_top_genes=2000)
    # Regress out the batch effect
    sc.pp.regress_out(adata, ['batch'])
    adata.X = scaler.inverse_transform(adata.X)
    plot_adata(adata)
    adata1_, adata2_ = get_batch_from_adata(adata)

    print(eval_mmd(torch.tensor(adata1_.X), torch.tensor(adata1.X)))
    print(eval_mmd(torch.tensor(adata2.X), torch.tensor(adata1_.X)))
