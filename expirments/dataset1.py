import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

# import sklearn
# from sklearn.preprocessing import MinMaxScaler
#
# from train2 import ber_for_notebook
# from unsupervised_utils import resample_data_normal, resample_data

from pre_procesing.train_reduce_dim import pre_processing


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_pca_result = (data - mean) / std

    return normalized_pca_result


def augment_unlabeled_data(data, num_augmented_samples=2000):
    augmented_data = []

    for _ in range(num_augmented_samples):
        # Randomly select an index from the original data
        index = np.random.randint(len(data))

        # Apply a simple data augmentation technique (e.g., random noise)
        augmented_sample = data[index] + np.random.normal(0, 0.05, size=data.shape[1])

        augmented_data.append(augmented_sample)

    augmented_data = np.array(augmented_data)
    return augmented_data


if __name__ == "__main__":
    config = {
        "lr": 0.001,  # Nuber of covariates in the data
        # or just tune.grid_search([<list of lists>])
        "dropout": 0.2,  # or tune.choice([<list values>])
        "weight_decay": 0.2,  # or tune.choice([<list values>])
        "batch_size": 500,  # or tune.choice([<list values>])
        "epochs": 500}

    data_dir = r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\new_data\dataset1"
    expression_data = os.path.join(data_dir, 'dataset1_sm_uc3.txt')
    adata = sc.read_text(expression_data, delimiter='\t', first_column_names=True, dtype='float64')
    adata = adata.transpose()
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_cells(adata, min_counts=5)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_genes(adata, min_counts=5)

    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    cell_info = os.path.join(data_dir, "sample_sm_uc3.txt")
    sample_adata = pd.read_csv(cell_info, header=0, index_col=0, sep='\t')

    adata.obs['batch'] = sample_adata.loc[adata.obs_names, "batch"]
    adata.obs['cell_type'] = sample_adata.loc[adata.obs_names, "celltype"]

    adata1 = adata[adata.obs['batch'] == 'Batch1', :].copy()
    adata2 = adata[adata.obs['batch'] == 'Batch2', :].copy()

    zero_columns = np.all(adata1.X == 0, axis=0)
    filtered_array_b1 = adata1.X[:, ~zero_columns]
    zero_columns = np.all(adata2.X == 0, axis=0)
    filtered_array_b2 = adata2.X[:, ~zero_columns]
    print(adata1.shape)
    source, target = pre_processing(adata1.X, adata2.X, num_epochs=400,
                                    save_weights_path=r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\weights\dataset1")

    min_max_scaler_src = MinMaxScaler()
    min_max_scaler_target = MinMaxScaler()
    normalized_src = min_max_scaler_src.fit_transform(source)
    normalized_target = min_max_scaler_target.fit_transform(target)
    # # normalized_src = resample_data(normalized_src,sample_size=2000)
    # # normalized_target = resample_data(normalized_target,sample_size=2000)
    # # source_df = normalize(src)
    # # target_df = normalize(adata2.obsm['X_pca'])
    # ber_for_notebook(normalized_src, normalized_target, config)
