import itertools
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from expirments.load import get_batch_from_adata
from plot import get_pca_data, plot_scatter
import scanpy as sc
def plot_umap_celltype(adata,path,title):
    num_pcs = 20
    sc.pp.neighbors(adata, n_pcs=num_pcs, n_neighbors=20)
    sc.tl.umap(adata)
    color_group = ['celltype']
    sc.pl.umap(adata, color=color_group)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(path,f'{title}_celltype.png'))


def plot_umap_batch(adata,path,title):
    num_pcs = 20
    sc.pp.neighbors(adata, n_pcs=num_pcs, n_neighbors=20)
    sc.tl.umap(adata)# , svd_solver='arpack', n_comps=5)
    color_group = ['batch']
    sc.pl.umap(adata, color=color_group)
    plt.savefig(os.path.join(path,f'{title}_batch.png'))

def plot_adata(adata, plot_dir='', embed='', label='celltype', title='before-calibrationp'):
#     adata1, adata2 = get_batch_from_adata(adata)
#     labels_b1 = adata1.obs[label]
#     labels_b2 = adata2.obs[label]
#     if embed != '':
#         src_pca = get_pca_data(adata1.obsm[embed])
#         target_pca = get_pca_data(adata2.obsm[embed])
#     else:
#         src_pca = get_pca_data(adata1.X)
#         target_pca = get_pca_data(adata2.X)
    plot_umap_batch(adata, path=plot_dir,title=title)
    plot_umap_celltype(adata, path=plot_dir,title=title)
    # plot_umap_batch(adata2, path=plot_dir)
    # plot_umap_celltype(adata2, path=plot_dir)
    # plot_scatter(src_pca, target_pca, labels_b1, labels_b2, plot_dir=plot_dir, title=title)


def make_combinations_from_config(config):
    param_combinations = list(itertools.product(*config.values()))
    configurations = []
    for params in param_combinations:
        new_config = {key: value for key, value in zip(config.keys(), params)}
        configurations.append(new_config)

    return configurations


def sample_from_space(configurations, num_of_samples):
    list_configurations = random.sample(configurations, num_of_samples)

    for index, config in enumerate(list_configurations):
        expirement_name = f"expirement_{index}"
        config["expirement_name"] = expirement_name
        config["plots_dir"] = os.path.join(config["plots_dir"], expirement_name)
        config["save_weights"] = os.path.join(config["save_weights"], expirement_name)

    return list_configurations
