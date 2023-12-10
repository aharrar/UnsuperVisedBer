import os

import numpy as np
import pandas as pd
from harmony import harmonize

from plot import get_pca_data, scatterHist
from expirments.load import load_and_pre_process_data
plots_dir = r"../plots/harmony/dataset-p2-3m"

os.makedirs(plots_dir,exist_ok=True)
path_src = '../../data/Cytof/Person2Day1_3month.csv'
path_target = '../../data/Cytof/Person2Day2_3month.csv'
src_path_label = '../../data/Cytof/Person2Day1_3month_label.csv'
target_path_label = '../../data/Cytof/Person2Day2_3month_label.csv'

if __name__ == "__main__":
    labels_b1 = np.loadtxt(src_path_label)
    labels_b2 = np.loadtxt(target_path_label)

    batch1_array, batch2_array = load_and_pre_process_data(path_src, path_target)
    src_pca = get_pca_data(batch1_array)
    target_pca = get_pca_data(batch2_array)

    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="before-calibration-labels",
                name1='target', name2='src', plots_dir=plots_dir)
    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="before-calibrationp-batch",
                name1='target', name2='src', to_plot_labels=False, plots_dir=plots_dir)

    X = np.vstack((batch1_array, batch2_array))

    n_cells_batch1 = len(batch1_array)
    n_cells_batch2 = len(batch2_array)
    df_metadata = pd.DataFrame({'Batch': ['Batch1'] * n_cells_batch1 + ['Batch2'] * n_cells_batch2})

    # Step 4: Use Harmony to harmonize the data
    # Specify 'Batch' as the batch key
    harmony_correction = harmonize(X, df_metadata, batch_key='Batch')
    corrected_batch_1 = harmony_correction[:n_cells_batch1]
    corrected_batch_2 = harmony_correction[n_cells_batch1:]
    src_pca = get_pca_data(corrected_batch_1)
    target_pca = get_pca_data(corrected_batch_2)

    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="after-calibration-labels",
                name1='target', name2='src', plots_dir=plots_dir)
    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title="after-calibrationp-batch",
                name1='target', name2='src', to_plot_labels=False, plots_dir=plots_dir)
