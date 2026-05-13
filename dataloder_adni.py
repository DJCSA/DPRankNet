from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torchio as tio


class Get_Dataset(Dataset):

    def __init__(self, data_path, transforms=None, double=None):
        self.subjects = pd.read_csv(data_path)
        self.index = list(self.subjects.columns.values)
        self.len_dataset = len(self.subjects[self.index[0]])
        self.index.pop(0)
        self.transforms = transforms
        self.double = double

    # self.index.pop(0)

    def __len__(self):
        return len(self.subjects[self.index[0]])

    def __getitem__(self, idx):
        img_path = self.subjects[self.index[0]][idx]
        img_path1 = img_path[31:-18]
        # print(img_path1)
        img_path2 = img_path[-11:]
        # print(img_path2)
        img_path = r"D:\DJC/dataset/" + img_path1 + img_path2
        img = np.load(img_path)
        assert img is not None
        group = self.subjects[self.index[1]][idx]

        img = img.astype(np.float32)
        tensor_data = torch.from_numpy(img)
        # print(tensor_data.shape)
        label = group
        if self.transforms is not None:
            tensor_data_aug = self.transforms(tensor_data)
            if self.double is not None:
                return tensor_data, tensor_data_aug, label, idx
            else:
                return tensor_data_aug, label
        else:
            return tensor_data, label


class Adni_dataloader():
    def __init__(self, data_path_train, data_path_val, transforms=None, double=None, batch_size=12, num_workers=8):
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.transforms = transforms
        self.double = double
        self.len_train_dataset = None
        self.len_val_dataset = None
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers
        if self.transforms is not None:
            self.data_aug = tio.Compose([  # tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),  # 随即翻转
                # tio.RandomAffine(scales=0.1, degrees=0, translation=0),
                tio.RandomAffine(scales=0, degrees=30, translation=5)
            ])

    def run(self):
        train_datasets = Get_Dataset(self.data_path_train, self.data_aug, self.double)
        self.len_train_dataset = train_datasets.len_dataset
        val_datasets = Get_Dataset(self.data_path_val)
        self.len_val_dataset = val_datasets.len_dataset

        train_dataloader = DataLoader(
            train_datasets,
            batch_size=self.batch_size,
            pin_memory=False,  # -
            num_workers=self.num_workers,
            shuffle=True,
            # collate_fn=Creat_AD_data.collate_fn,  # -
            drop_last=False
        )
        select_dataloader = DataLoader(
            train_datasets,
            batch_size=self.batch_size,
            pin_memory=False,  # -
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=Creat_AD_data.collate_fn,  # -
            drop_last=False
        )

        val_dataloader = DataLoader(
            val_datasets,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=self.num_workers,
            shuffle=False,
            #  collate_fn=Creat_AD_data.collate_fn,
            drop_last=False
        )

        return train_dataloader, val_dataloader, select_dataloader
