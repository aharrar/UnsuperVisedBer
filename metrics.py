import sklearn
import torch
from torch import nn


class MMD(nn.Module):
    def __init__(self,
                 src,
                 target,
                 target_sample_size=1000,
                 n_neighbors=25,
                 scales=None,
                 weights=None):
        super(MMD, self).__init__()
        if scales is None:
            med_list = torch.zeros(25)
            for i in range(25):
                sample = target[torch.randint(0, target.shape[0] - 1, (target_sample_size,))]
                distance_matrix = torch.cdist(sample, sample)
                sorted, indices = torch.sort(distance_matrix, dim=0)

                # nearest neighbor is the point so we need to exclude it
                med_list[i] = torch.median(sorted[:, 1:n_neighbors])
            med = torch.mean(med_list)

        scales = [med / 2, med, med * 2]  # CyTOF

        # print(scales)
        scales = torch.tensor(scales)
        weights = torch.ones(len(scales))
        self.src = src
        self.target = target
        self.target_sample_size = target_sample_size
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    def RaphyKernel(self, X, Y):
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = (torch.cdist(X, Y) ** 2).unsqueeze(0)
        scales = self.scales.unsqueeze(-1).unsqueeze(-1)
        weights = self.weights.unsqueeze(-1).unsqueeze(-1)

        return torch.sum(weights * torch.exp(-sQdist / (torch.pow(scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self):
        mmd_epoch_list = torch.zeros(3)
        for index in range(3):
            mmd_list = torch.zeros(25)
            for i in range(25):
                src = self.src[torch.randint(0, self.src.shape[0] - 1, (self.target_sample_size,))]
                target = self.target[torch.randint(0, self.target.shape[0] - 1, (self.target_sample_size,))]
                xx = self.kernel(src, src)
                xy = self.kernel(src, target)
                yy = self.kernel(target, target)
                # calculate the bias MMD estimater (cannot be less than 0)
                MMD = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
                mmd_list[i] = torch.sqrt(MMD)
            mmd_epoch_list[index] = torch.mean(mmd_list)
            # return the square root of the MMD because it optimizes better
        return torch.mean(mmd_epoch_list)


def eval_mmd(source, target):
    mmd_value = MMD(source, target).cost()

    return mmd_value


import numpy as np
from sklearn.metrics import pairwise_distances

from sklearn.metrics import silhouette_score


def silhouette(adata, group_key='batch', metric='euclidean', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating
    overlapping clusters and -1 indicating misclassified cells
    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    """
    asw = silhouette_score(
        X=adata.X,
        labels=adata.obs[group_key],
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


import torch
import numpy as np


def batch_kl(adata, batch_key='batch'):
    """
    Calculate the BatchKL metric for batch correction.

    Args:
        meta_data (numpy.ndarray): Meta data (e.g., batch information).
        embedding (torch.Tensor): Low-dimensional embeddings (e.g., PCA, t-SNE, UMAP).

    Returns:
        float: BatchKL metric.
    """
    # Assuming meta_data contains batch information (e.g., batch labels)
    # and embedding is a torch.Tensor with shape (n_cells, n_features)

    # Calculate mean and variance for each batch
    batch_means = []
    batch_vars = []
    adata1 = adata[adata.obs[batch_key] == 1, :].copy()
    adata2 = adata[adata.obs[batch_key] == 2, :].copy()

    unique_batches = np.unique(adata.obs[batch_key])
    for batch in unique_batches:
        batch_indices = np.where(adata.obs[batch_key] == batch)[0]
        batch_embedding = adata[batch_indices].X
        batch_mean = np.mean(batch_embedding, axis=0)
        batch_var = np.var(batch_embedding, axis=0)
        batch_means.append(batch_mean)
        batch_vars.append(batch_var)

    # Compute BatchKL
    batch_kl_sum = 0.0
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            kl_divergence = np.sum(
                0.5 * (batch_vars[i] / batch_vars[j] + (batch_means[j] - batch_means[i]) ** 2 / batch_vars[
                    j] - 1.0 + np.log(batch_vars[j] / batch_vars[i]))
            )
            batch_kl_sum += kl_divergence

    # Normalize by the number of batch pairs
    num_batch_pairs = len(unique_batches) * (len(unique_batches) - 1) // 2
    batch_kl = batch_kl_sum / num_batch_pairs

    return batch_kl


def compute_kbet(adata, batch_key='batch'):
    """
    Compute the kBET score for batch effect assessment.

    Args:
        adata (AnnData): Annotated data matrix.
        batch_key (str): Column name in adata.obs containing batch information.

    Returns:
        float: kBET score.
    """
    # Extract batch labels and cell embeddings
    batch_labels = adata.obs[batch_key]
    cell_embeddings = adata.X

    # Compute pairwise distances between cells
    dist_matrix = pairwise_distances(cell_embeddings, metric='euclidean')

    # Calculate average distance within each batch
    intra_batch_distances = []
    for batch_id in np.unique(batch_labels):
        batch_indices = np.where(batch_labels == batch_id)[0]
        batch_distances = dist_matrix[batch_indices][:, batch_indices]
        intra_batch_distances.append(np.mean(batch_distances))

    # Compute average distance between batches
    inter_batch_distances = np.mean(intra_batch_distances)

    # Calculate kBET score
    k_bet_score = inter_batch_distances / np.mean(intra_batch_distances)

    return k_bet_score


import scanpy as sc


# Find optimal resolution given ncluster
def find_resolution(adata_, n_clusters, random):
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions) / 2
        sc.tl.louvain(adata, resolution=current_res, random_state=random)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        iteration = iteration + 1
    return current_res


def calulate_ari_nmi(adata_integrated, n_cluster=4):
    sc.pp.neighbors(adata_integrated, random_state=0)
    reso = find_resolution(adata_integrated, n_cluster, 0)
    sc.tl.louvain(adata_integrated, reso, random_state=0)
    sc.tl.umap(adata_integrated)
    if (adata_integrated.X.shape[1] == 2):
        adata_integrated.obsm["X_emb"] = adata_integrated.X
    #         sc.pl.embedding(adata_integrated, basis='emb', color = ['louvain'], wspace = 0.5)
    #     else:
    #         sc.pl.umap(adata_integrated,color=["louvain"])

    ARI = ari(adata_integrated.obs["celltype"].astype(str), adata_integrated.obs["louvain"])
    NMI = normalized_mutual_info_score(adata_integrated.obs["celltype"].astype(str), adata_integrated.obs["louvain"])
    print("louvain clustering result(resolution={}):n_cluster={}".format(reso, n_cluster))
    print("ARI:", ARI)
    print("NMI:", NMI)
    return ARI, NMI
