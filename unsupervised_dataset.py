import torch
from torch.utils.data import Dataset


def organize_data(src_data, dest_data,src_labels=None,target_labels=None):
    data_set = []

    if src_labels is None or target_labels == None:
        [data_set.append((src,None, torch.tensor(0, dtype=torch.float32))) for src in src_data]
        [data_set.append((dest,None, torch.tensor(1., dtype=torch.float32))) for dest in dest_data]
    else:
        [data_set.append((src,src_labels[index], torch.tensor(0, dtype=torch.float32))) for index,src in enumerate(src_data)]
        [data_set.append((dest,target_labels[index], torch.tensor(1., dtype=torch.float32))) for index,dest in enumerate(dest_data)]

    return data_set


class UnsupervisedDataset(Dataset):
    def __init__(self, src_data, dest_data,src_labels=None,target_labels=None):
        self.src_data = src_data
        self.dest_data = dest_data
        self.data = organize_data(src_data, dest_data,src_labels,target_labels)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_item = self.data[idx]

        return data_item
