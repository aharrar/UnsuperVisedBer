import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from sklearn.preprocessing import MinMaxScaler

from expirments.load import make_adata_from_batches, assign_labels_to_numbers, get_batch_from_adata
from expirments.utils import sample_from_space, make_combinations_from_config, plot_adata
from main import ber_for_notebook
from plot import get_pca_data, plot_data
from pre_procesing.train_reduce_dim import pre_processing
from scDML.scDML.metrics import evaluate_dataset

from scDML.scDML.metrics import evaluate_dataset, silhouette_coeff_ASW

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "lr": [0.01],#  np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [ 0.2,0.35,0.3],  # or tune.choice([<list values>])
    "batch_size": [ 128,64,36],  # or tune.choice([<list values>])
    "epochs": [200],
    "coef_1": [100,125,150],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset2-benchmark/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset2-benchmark/")]
}

dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")

configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)

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
    zero_columns = np.all(adata.X == 0, axis=0)
    filtered_array_b1 = adata1.X[:, ~zero_columns]
    # zero_columns = np.all(adata2.X == 0, axis=0)
    filtered_array_b2 = adata2.X[:, ~zero_columns]

    source = filtered_array_b1
    target = filtered_array_b2

    os.makedirs(dim_reduce_weights_path, exist_ok=True)
    source, target,model_shrinking = pre_processing(source, target, num_epochs=50,
                                    save_weights_path=dim_reduce_weights_path)

    adata_dim_reduce = sc.AnnData(X=np.concatenate((source, target), axis=0))
    adata_dim_reduce.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    adata_dim_reduce.obs['batch'] = [1 if i < len(source) else 2 for i in range(len(source) + len(target))]
    sc.tl.pca(adata_dim_reduce, svd_solver='arpack', n_comps=20)
    adata_dim_reduce.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    print("----before----")
    # evaluate_dataset(adata_dim_reduce)
    adata1 = adata_dim_reduce[adata_dim_reduce.obs['batch'] == 1, :].copy()
    adata2 = adata_dim_reduce[adata_dim_reduce.obs['batch'] == 2, :].copy()

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata_dim_reduce, plot_dir=config["plots_dir"], title='before-calibrationp')
        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))
        adata1, adata2 = get_batch_from_adata(adata_dim_reduce)
        adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                    adata2=adata2)
        adata_target_calibrated_src.X = np.array(model_shrinking.decoder(adata_target_calibrated_src.X))
        adata_src_calibrated_target.X = np.array(model_shrinking.decoder(adata_src_calibrated_target.X))

        plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')
        plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
                   title='after-calibration-src_calibrated_target')

        evaluate_dataset(adata_src_calibrated_target).to_csv(
            os.path.join(config["plots_dir"], "adata_src_calibrated_target.csv"))  # Set the batch key for each cell
        evaluate_dataset(adata_target_calibrated_src).to_csv(
            os.path.join(config["plots_dir"], "adata_target_calibrated_src.csv"))  # Set the batch key for each cell
