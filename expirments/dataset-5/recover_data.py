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

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {"lr": 0.01, "dropout": 0.2, "weight_decay": 0.2, "batch_size": 36, "epochs": 200, "coef_1": 100, "save_weights": r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\expirments\weights\ber\dataset5-benchmark", "plots_dir": "C:\\Users\\avrah\\PycharmProjects\\UnsuperVisedBer\\expirments\\plots/ours/dataset5-TRY12/expirement_0", "expirement_name": "expirement_0"}

if __name__ == "__main__":
    weight_path = config["save_weights"]
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset5"
    dim_reduce_weights_path = os.path.join(weight_path, "dim_reduce")
    load_weights_path = os.path.join(dim_reduce_weights_path, "weights.pt")

    adata = sc.read_h5ad(os.path.join(data_dir, 'myTotalData_scale_with_pca.h5ad'))
    adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['cell_type']))

    adata1 = adata[adata.obs['batch'] == 2, :].copy()
    adata2 = adata[adata.obs['batch'] == 1, :].copy()
    source, target, model_for_shrinking_data = pre_processing(adata1.X, adata2.X, num_epochs=20,
                                                              load_weights_path=dim_reduce_weights_path)
    adata1.obsm["dim_reduce"], adata2.obsm["dim_reduce"] = source, target

    load_weights_path = os.path.join(weight_path, "expirement_2")
    adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                adata2=adata2,
                                                                                load_pre_weights=load_weights_path,
                                                                                embed='dim_reduce')

    # os.makedirs(config["save_weights"], exist_ok=True)
    os.makedirs(config["plots_dir"], exist_ok=True)

    plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
               title='after-calibration-target_calibrated_src')
    plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
               title='after-calibration-src_calibrated_target')

    # print(eval_mmd(torch.tensor(adata1.obsm["X_pca"]), torch.tensor(adata2.obsm["X_pca"])))

    # scaler = MinMaxScaler()
    # # Scale gene values between zero and one
    # t_adata.X = scaler.fit_transform(t_adata.X)

    # plot_adata(adata_target_calibrated_src)
    # plot_adata(adata_src_calibrated_target)

    # calibrated_data = torch.tensor(adata_target_calibrated_src.X)
    # calibrated_data = model_for_shrinking_data.decoder(calibrated_data)
    # adata.X = calibrated_data.detach().numpy()
    # plot_adata(adata)
