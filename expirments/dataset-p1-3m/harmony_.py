import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from harmony import harmonize
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize

from expirments.load import load_and_pre_process_data, load_to_adata_shaham_dataset, get_batch_from_adata, \
    make_adata_from_batches
from expirments.utils import plot_adata
from metrics import eval_mmd
from plot import get_pca_data, scatterHist
# from scDML.scDML.metrics import evaluate_dataset, silhouette_coeff_ASW

# from scDML.scDML.metrics import evaluate_dataset
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer')
plots_dir = os.path.join(parent_dir, r"/plots/harmony/dataset-p1-3m")

os.makedirs(plots_dir, exist_ok=True)
path_src = os.path.join(parent_dir, 'data/Cytof/Person1Day1_3month.csv')
path_target = os.path.join(parent_dir, 'data/Cytof/Person1Day2_3month.csv')
src_path_label = os.path.join(parent_dir, 'data/Cytof/Person1Day1_3month_label.csv')
target_path_label = os.path.join(parent_dir, 'data/Cytof/Person1Day2_3month_label.csv')

import scanpy as sc

if __name__ == "__main__":
    labels_b1 = np.loadtxt(src_path_label)
    labels_b2 = np.loadtxt(target_path_label)

    adata = load_to_adata_shaham_dataset(path_src, path_target, src_path_label, target_path_label)
    adata.X = normalize(torch.tensor(adata.X), p=2, dim=1)

    # evaluate_dataset(adata)
    # silhouette_coeff_ASW(adata)
    adata1, adata2 = get_batch_from_adata(adata)
    batch1_array, batch2_array = adata1.X, adata2.X

    # evaluate_dataset(adata)

    plot_adata(adata, plot_dir=plots_dir, title='after-calibration')

    X = np.vstack((batch1_array, batch2_array))

    n_cells_batch1 = len(batch1_array)
    n_cells_batch2 = len(batch2_array)
    df_metadata = pd.DataFrame({'Batch': ['Batch1'] * n_cells_batch1 + ['Batch2'] * n_cells_batch2})

    # Step 4: Use Harmony to harmonize the data
    # Specify 'Batch' as the batch key
    harmony_correction = harmonize(X, df_metadata, batch_key='Batch')

    from harmony import harmonize
    Z = harmonize(adata.X, adata.obs, batch_key = 'batch')
    adata.X = Z
    adata1, adata2 = get_batch_from_adata(adata)

    # corrected_batch_1 = harmony_correction[:n_cells_batch1]
    # corrected_batch_2 = harmony_correction[n_cells_batch1:]

    # adata = make_adata_from_batches(corrected_batch_1, corrected_batch_2, labels_b1, labels_b2)
    # sc.pp.normalize_total(adata)
    print(eval_mmd(torch.tensor( adata1.X), torch.tensor(adata2.X)))

    plot_adata(adata, plot_dir=plots_dir,
               title='after-calibration')
    # evaluate_dataset(adata).to_csv(
    #         os.path.join(plots_dir, "metrics.csv"))
    # silhouette_coeff_ASW(adata)
