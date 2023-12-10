import os

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from sklearn.preprocessing import MinMaxScaler

from main import ber_for_notebook
from plot import get_pca_data, plot_data
from pre_procesing.train_reduce_dim import pre_processing
from scDML.scDML.metrics import evaluate_dataset

config = {
    "lr": 0.01,  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": 0.2,  # or tune.choice([<list values>])
    "weight_decay": 0.2,  # or tune.choice([<list values>])
    "batch_size": 128,  # or tune.choice([<list values>])
    "epochs": 500,
    "save_weights": r"/weights/ber/dataset2/",
    "plots_dir": r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\plots\dataset2"}

os.makedirs(config["save_weights"], exist_ok=True)
os.makedirs(config["plots_dir"], exist_ok=True)

if __name__ == "__main__":
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\BER\new_data\dataset-2"

    expr_filename = os.path.join(data_dir, 'filtered_total_batch1_seqwell_batch2_10x.txt')

    adata = sc.read_text(expr_filename, delimiter='\t', first_column_names=True, dtype='float64')
    adata = adata.T

    # Read sample info
    metadata_filename = os.path.join(data_dir, "filtered_total_sample_ext_organ_celltype_batch.txt")
    sample_adata = pd.read_csv(metadata_filename, header=0, index_col=0, sep='\t')

    adata.obs['batch'] = sample_adata.loc[adata.obs_names, "batch"]
    adata.obs['celltype'] = sample_adata.loc[adata.obs_names, "orig.ident"]

    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_cells(adata, min_counts=5)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_genes(adata, min_counts=5)
    # sc.pp.scale(adata)
    sc.pp.log1p(adata)

    # Set the batch key for each cell

    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()

    zero_columns = np.all(adata1.X == 0, axis=0)
    filtered_array_b1 = adata1.X[:, ~zero_columns]
    zero_columns = np.all(adata2.X == 0, axis=0)
    filtered_array_b2 = adata2.X[:, ~zero_columns]

    source = filtered_array_b1
    target = filtered_array_b2
    plot_data(source, target, save_dir=config["plots_dir"])
    source, target = pre_processing(source, target, num_epochs=200,
                                    load_weights_path=r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\weights\dim-reduction\dataset2\weights.pt")

    adata_dim_reduce = sc.AnnData(X=np.concatenate((source, target), axis=0))
    adata_dim_reduce.obs['celltype'] = adata.obs['celltype'].to_numpy()
    adata_dim_reduce.obs['batch'] = [1 if i < len(source) else 2 for i in range(len(source) + len(target))]
    print("----before----")
    evaluate_dataset(adata_dim_reduce)

    adata_src_calibrated_target, adata_target_calibrated_src = ber_for_notebook(source, target
                                                                                , config)
    adata_src_calibrated_target.obs['celltype'] = np.concatenate(
        (adata1.obs['celltype'].to_numpy(), adata2.obs['celltype']), axis=0)
    adata_target_calibrated_src.obs['celltype'] = np.concatenate(
        (adata2.obs['celltype'].to_numpy(), adata1.obs['celltype']), axis=0)
    print("-----after-------")
    evaluate_dataset(adata_src_calibrated_target)
    evaluate_dataset(adata_target_calibrated_src)
