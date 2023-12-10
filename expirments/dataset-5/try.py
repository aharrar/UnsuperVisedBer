import os
from pathlib import Path

import numpy as np
import scanpy as sc

from expirments.load import assign_labels_to_numbers
from scDML.scDML.metrics import silhouette_coeff_ASW
from unsupervised.autoencoder import Encoder
from unsupervised.ber_network import DecoderBer, Net

data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset5"
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')
weights_net = os.path.join(parent_dir, "weights/ber/dataset5-benchmark/expirement_0")

adata = sc.read_h5ad(os.path.join(data_dir, 'myTotalData_scale_with_pca.h5ad'))
adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['cell_type']))


silhouette_coeff_ASW(adata)
# input_dim = adata.X.shape[1]
# code_dim = 25
# encoder = Encoder(input_dim,
#                   hidden_dim=20,
#                   drop_prob=0.1,
#                   code_snape=code_dim)
#
# decoder = DecoderBer(code_dim=code_dim,
#                      hidden_dim=100,
#                      output_dim=input_dim,
#                      drop_prob=0.1,
#                      )
# net = Net(encoder, decoder)
# net.from_pretrain(weights_net)
