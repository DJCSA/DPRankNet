from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch


class Get_Dataset(Dataset):

    def __init__(self, data_path, AIBL):
        self.subjects = pd.read_csv(data_path)
        self.index = list(self.subjects.columns.values)
        self.len_dataset = len(self.subjects[self.index[0]])
        self.index.pop(0)
        self.AIBL = AIBL

    def __len__(self):
        return len(self.subjects[self.index[0]])

    def __getitem__(self, idx):
        img_path = self.subjects[self.index[0]][idx]
        if self.AIBL is True:
            group = self.subjects[self.index[1]][idx]
            # group = group.long()
            if group == 0:
                img_path = r"D:\DJC\AIBL\AD/" + img_path
            elif group == 1:
                img_path = r"D:\DJC\AIBL\MCI/" + img_path
            else:
                img_path = r"D:\DJC\AIBL\NC/" + img_path
            img = np.load(img_path)
            assert img is not None
            group = self.subjects[self.index[1]][idx]
            img = img.astype(np.float32)
            tensor_data = torch.from_numpy(img)
            tensor_data = tensor_data.unsqueeze(0)
            label = group
        else:
            img_path1 = img_path[23:-18]
            img_path2 = img_path[-11:]
            img_path = r"D:\DJC/" + img_path1 + img_path2
            # print(img_path)
            # print(idx)
            img = np.load(img_path)
            assert img is not None
            group = self.subjects[self.index[1]][idx]
            img = img.astype(np.float32)
            tensor_data = torch.from_numpy(img)
            label = group

        return tensor_data, label, idx


class Adni_dataloader():
    def __init__(self, data_path_test, batch_size=12, num_workers=8, AIBL=None):
        self.data_path_test = data_path_test
        self.len_test_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.AIBL = AIBL

    def run(self):
        test_datasets = Get_Dataset(self.data_path_test, self.AIBL)
        self.len_test_dataset = test_datasets.len_dataset

        test_dataloader = DataLoader(
            test_datasets,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False
        )

        return test_dataloader
