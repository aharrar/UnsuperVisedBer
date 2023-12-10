import os
from pathlib import Path
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from expirments.load import assign_labels_to_numbers, make_adata_from_batches, get_batch_from_adata
from expirments.utils import plot_adata
from main import get_data_calibrated, ber_for_notebook
from metrics import eval_mmd
from pre_procesing.ae_for_shrink_dim import AutoencoderShrink
from pre_procesing.train_reduce_dim import pre_processing
from unsupervised.autoencoder import Encoder
from unsupervised.ber_network import DecoderBer, Net


def get_dataset_2():
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\BER\new_data\dataset-2"

    # expr_filename = os.path.join(data_dir, 'filtered_total_batch1_seqwell_batch2_10x.txt')
    #
    # adata = sc.read_text(expr_filename, delimiter='\t', first_column_names=True, dtype='float64')
    # adata = adata.T
    #
    # # Read sample info
    # metadata_filename = os.path.join(data_dir, "filtered_total_sample_ext_organ_celltype_batch.txt")
    # sample_adata = pd.read_csv(metadata_filename, header=0, index_col=0, sep='\t')
    #
    # adata.obs['batch'] = sample_adata.loc[adata.obs_names, "batch"]
    # adata.obs['celltype'] = sample_adata.loc[adata.obs_names, "orig.ident"]
    #
    # sc.pp.filter_cells(adata, min_genes=300)
    # sc.pp.filter_cells(adata, min_counts=5)
    # sc.pp.filter_genes(adata, min_cells=5)
    # sc.pp.filter_genes(adata, min_counts=5)
    # # sc.pp.scale(adata)
    # sc.pp.log1p(adata)
    adata = sc.read_h5ad(os.path.join(data_dir, 'myTotalData_scale_with_pca.h5ad'))
    # Set the batch key for each cell
    adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()
    zero_columns = np.all(adata.X == 0, axis=0)
    adata1.X = adata1.X[:, ~zero_columns]
    # zero_columns = np.all(adata2.X == 0, axis=0)
    adata2.X = adata2.X[:, ~zero_columns]
    return adata, adata1, adata2

config = {"lr": 0.01, "dropout": 0.2, "weight_decay": 0.2, "batch_size": 36, "epochs": 200, "coef_1": 100, "expirement_name": "expirement_0"}

if __name__ == "__main__":
    parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')
    weight_path = os.path.join(parent_dir, "weights/ber/dataset2-benchmark/")

    dim_reduce_weights_path = os.path.join(weight_path, "dim_reduce")

    load_weights_path = os.path.join(dim_reduce_weights_path, "weights.pt")
    adata, adata1, adata2 = get_dataset_2()
    source, target, model_for_shrinking_data = pre_processing(adata1.X, adata2.X, num_epochs=20,
                                                              load_weights_path=dim_reduce_weights_path)

    load_weights_path = os.path.join(weight_path, "expirement_3")
    adata1.obsm["dim_reduce"], adata2.obsm["dim_reduce"] = source, target

    adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                adata2=adata2,
                                                                                load_pre_weights=load_weights_path,
                                                                                embed='dim_reduce')

    # os.makedirs(config["save_weights"], exist_ok=True)

    plot_adata(adata_target_calibrated_src)
    plot_adata(adata_src_calibrated_target)


    # scaler = MinMaxScaler()
    # # Scale gene values between zero and one
    # t_adata.X = scaler.fit_transform(t_adata.X)


    # plot_adata(adata_target_calibrated_src)
    # plot_adata(adata_src_calibrated_target)

    # calibrated_data = torch.tensor(adata_target_calibrated_src.X)
    # calibrated_data = model_for_shrinking_data.decoder(calibrated_data)
    # adata.X = calibrated_data.detach().numpy()
    # plot_adata(adata)
