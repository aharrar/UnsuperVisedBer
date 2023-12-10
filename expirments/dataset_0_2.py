from sklearn.preprocessing import MinMaxScaler
from ZIFA import ZIFA
from ZIFA import block_ZIFA
from plot_data import scatterHist, get_pca_data
from pre_prosessing.pre_processing import load_and_pre_process_data, pre_processing
from train2 import ber_for_notebook
config = {
        "lr": 0.01,  # Nuber of covariates in the data
        # or just tune.grid_search([<list of lists>])
        "dropout": 0.2,  # or tune.choice([<list values>])
        "weight_decay": 0.2,  # or tune.choice([<list values>])
        "batch_size": 128,  # or tune.choice([<list values>])
        "epochs": 500}

path_src = '/home/aharrar/PycharmProjects/BER/data/Person2Day1_baseline.csv'
path_target = '/home/aharrar/PycharmProjects/BER/data/Person2Day2_baseline.csv'

src_data_without_labels, target_data_without_labels = load_and_pre_process_data(path_src, path_target)


min_max_scaler_src = MinMaxScaler((-0.7,0.7))
min_max_scaler_target = MinMaxScaler((-0.7,0.7))

normalized_src = min_max_scaler_src.fit_transform(src_data_without_labels)
normalized_target = min_max_scaler_target.fit_transform(target_data_without_labels)

ber_for_notebook(normalized_src, normalized_target,config)
