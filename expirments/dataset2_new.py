import numpy as np
from sklearn.preprocessing import MinMaxScaler

from train2 import ber_for_notebook

src_path = "/home/aharrar/PycharmProjects/BER/new_data/dataset2/dataset3-batch1.csv"
target_path = "/home/aharrar/PycharmProjects/BER/new_data/dataset2/dataset3-batch2.csv"

src_data = np.loadtxt(src_path)
target_data = np.loadtxt(target_path)
min_max_scaler_src = MinMaxScaler((0,10))
min_max_scaler_target = MinMaxScaler((0,10))

normalized_src = min_max_scaler_src.fit_transform(src_data)
normalized_target = min_max_scaler_target.fit_transform(target_data)

ber_for_notebook(normalized_src, normalized_target,config)
