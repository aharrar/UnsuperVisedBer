# Import the necessary packages
import os
from pathlib import Path

import scanpy as sc
import numpy as np

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize

from expirments.load import load_and_pre_process_data, load_to_adata_shaham_dataset, get_batch_from_adata, \
    assign_labels_to_numbers
from expirments.utils import plot_adata
from metrics import eval_mmd
from plot import get_pca_data, scatterHist
from pre_procesing.ae_for_shrink_dim import AutoencoderShrink


# from scDML.scDML.metrics import evaluate_dataset

def get_dataset_2():
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\BER\new_data\dataset-2"
    adata = sc.read_h5ad(os.path.join(data_dir, 'myTotalData_scale_with_pca.h5ad'))
    # Set the batch key for each cell
    adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    zero_columns = np.all(adata.X == 0, axis=0)
    adata.X = adata.X[:, ~zero_columns]
    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()
    sc.pp.filter_cells(adata, min_genes=1)
    # zero_columns = np.all(adata2.X == 0, axis=0)
    # adata2.X = adata2.X[:, ~zero_columns]
    return adata, adata1, adata2

if __name__ == "__main__":
    parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

    adata, adata1, adata2 = get_dataset_2()
    sc.pp.normalize_total(adata)
    batch1, batch2 = get_batch_from_adata(adata)

    weight_path = os.path.join(parent_dir, "weights/ber/dataset2-benchmark/")

    dim_reduce_weights_path = os.path.join(weight_path, "dim_reduce")

    load_weights_path = os.path.join(dim_reduce_weights_path, "weights.pt")
    input_dim = adata.X.shape[1]
    ae_encoding_dim = 25
    model_for_shrinking_data = AutoencoderShrink(input_dim, ae_encoding_dim, hidden_layers=6)
    # train_data_ae_tensor = torch.cat((src_data, target_data), dim=0)
    model_for_shrinking_data.from_pretrain(load_weights_path)
    model_for_shrinking_data.eval()
    model_for_shrinking_data.encoder.eval()
    model_for_shrinking_data.decoder.eval()

    datareduce = model_for_shrinking_data.encoder(torch.tensor(adata.X).float()).detach().numpy()
    t_adata = sc.AnnData(X=datareduce)
    t_adata.obs['batch'] = np.array(adata.obs["batch"])
    t_adata.obs['celltype'] = adata.obs["celltype"]
    scaler = MinMaxScaler()
    # Scale gene values between zero and one
    t_adata.X = scaler.fit_transform(t_adata.X)

    # sc.pp.scale(t_adata, zero_center=True, max_value=1)    # sc.pp.normalize_total(t_adata)
    sc.pp.filter_cells(t_adata, min_genes=1)

    # Identify highly variable genes using the Seurat flavor
    sc.pp.highly_variable_genes(t_adata, batch_key='batch', flavor='seurat_v3', n_top_genes=3000)
    # Regress out the batch effect
    sc.pp.regress_out(t_adata, ['batch'])
    t_adata.X = scaler.inverse_transform(t_adata.X)
    plot_adata(t_adata)
    adata1, adata2 = get_batch_from_adata(t_adata)

    print(eval_mmd(torch.tensor(adata1.X), torch.tensor(adata2.X)))
