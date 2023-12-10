# Import the necessary packages
import scanpy as sc
import numpy as np

import torch
from sklearn.preprocessing import MinMaxScaler

from expirments.load import load_and_pre_process_data
from metrics import eval_mmd
from plot import get_pca_data, scatterHist

path_src = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_baseline.csv'
path_target = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_baseline.csv'

src_data_without_labels, target_data_without_labels = load_and_pre_process_data(path_src, path_target)

# min_max_scaler_src = MinMaxScaler((-0.7, 0.7))
# min_max_scaler_target = MinMaxScaler((-0.7, 0.7))
#
# src_data_without_labels = min_max_scaler_src.fit_transform(src_data_without_labels)
# target_data_without_labels = min_max_scaler_target.fit_transform(target_data_without_labels)

batch1 = src_data_without_labels
batch2 = target_data_without_labels
# Define the two numpy arrays
# Combine them into one expression matrix
print(f"mmd before {eval_mmd(torch.tensor(batch1), torch.tensor(batch2))}")
src_pca = get_pca_data(batch1)
target_pca = get_pca_data(batch2)

# scatterHist(target_pca[:, 0],
#             target_pca[:, 1],
#             src_pca[:, 0],
#             src_pca[:, 1],
#             "pc1", "pc2", title="train data after calibration",
#             name1='src', name2='target', plots_dir='')

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

adata1 = adata[adata.obs['batch'] == 1, :].copy()
adata2 = adata[adata.obs['batch'] == 2, :].copy()

src = adata1.X
target = adata2.X
src_pca = get_pca_data(src)
target_pca = get_pca_data(target)

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
