# Import the necessary packages
import os

import scanpy as sc
import numpy as np

import torch
from sklearn.preprocessing import MinMaxScaler

from expirments.load import load_and_pre_process_data
from metrics import eval_mmd
from plot import get_pca_data, scatterHist

plots_dir = r"../plots/seurat/dataset-p2-3m"

os.makedirs(plots_dir, exist_ok=True)

path_src = '../../data/Cytof/Person2Day1_3month.csv'
path_target = '../../data/Cytof/Person2Day2_3month.csv'
src_path_label = '../../data/Cytof/Person2Day1_3month_label.csv'
target_path_label = '../../data/Cytof/Person2Day2_3month_label.csv'

if __name__ == "__main__":
    src_data_without_labels, target_data_without_labels = load_and_pre_process_data(path_src, path_target)

    labels_b1 = np.loadtxt(src_path_label)
    labels_b2 = np.loadtxt(target_path_label)

    batch1 = src_data_without_labels
    batch2 = target_data_without_labels
    # Define the two numpy arrays
    # Combine them into one expression matrix
    # print(f"mmd before {eval_mmd(torch.tensor(batch1), torch.tensor(batch2))}")
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

    # Load your data into an AnnData object
    adata = sc.AnnData(X=np.concatenate((batch1, batch2), axis=0))
    # Set the batch key for each cell
    adata.obs['batch'] = ['batch1' if i < len(batch1) else 'batch2' for i in range(len(batch1) + len(batch2))]

    sc.pp.normalize_total(adata)

    # Identify highly variable genes using the Seurat flavor
    sc.pp.highly_variable_genes(adata, batch_key='batch', flavor='seurat')
    # Regress out the batch effect
    sc.pp.regress_out(adata, ['batch'])
    # Separate the data back into two batches
    # print(f"mmd {eval_mmd(torch.tensor(adata.X), torch.tensor(src_data_without_labels))}")
    # print(f"mmd {eval_mmd(torch.tensor(adata.X), torch.tensor(target_data_without_labels))}")

    adata1 = adata[adata.obs['batch'] == 'batch1', :].copy()
    adata2 = adata[adata.obs['batch'] == 'batch2', :].copy()

    src = adata1.X
    target = adata2.X
    src_pca = get_pca_data(src)
    target_pca = get_pca_data(target)
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

#
# scatterHist(batch1_corrected_pca[:, 0],
#             batch1_corrected_pca[:, 1],
#             src_pca[:, 0],
#             src_pca[:, 1],
#             "pc1", "pc2", title="train data after calibration",
#             name1='target', name2='calibrated', plots_dir='')
# batch2_corrected_pca = get_pca_data(batch_2)
# target_pca = get_pca_data(target)
#
# scatterHist(batch2_corrected_pca[:, 0],
#             batch2_corrected_pca[:, 1],
#             target_pca[:, 0],
#             target_pca[:, 1],
#             "pc1", "pc2", title="train data after calibration",
#             name1='target', name2='calibrated', plots_dir='')
# print(f"mmd after {eval_mmd(torch.tensor(batch_1), torch.tensor(src))}")
# print(f"mmd after {eval_mmd(torch.tensor(batch_2), torch.tensor(target))}")
#
#
#
