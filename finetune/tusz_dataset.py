import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, num_channels=19, num_patches=30, patch_size=200, task_type='binaryclass'):
        super().__init__()
        self.files = [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith('.pkl')
        ]
        self.num_channels = int(num_channels)
        self.num_patches = int(num_patches)
        self.patch_size = int(patch_size)
        self.task_type = str(task_type).lower()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_dict = pickle.load(open(self.files[idx], 'rb'))
        data_key = 'X' if 'X' in data_dict else 'signal'
        label_key = 'y' if 'y' in data_dict else 'label'
        data = np.asarray(data_dict[data_key], dtype=np.float32).reshape(
            self.num_channels, self.num_patches, self.patch_size
        )
        raw_label = np.asarray(data_dict[label_key]).reshape(-1)[0]
        label = int(raw_label) if self.task_type == 'multiclass' else float(raw_label)
        return data / 100.0, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch], dtype=np.float32)
        if self.task_type == 'multiclass':
            y_label = np.array([x[1] for x in batch], dtype=np.int64)
        else:
            y_label = np.array([x[1] for x in batch], dtype=np.float32)
        return torch.from_numpy(x_data), torch.from_numpy(y_label)


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(
            os.path.join(self.datasets_dir, 'processed_train'),
            num_channels=self.params.num_channels,
            num_patches=self.params.num_patches,
            patch_size=self.params.patch_size,
            task_type=self.params.task_type,
        )
        val_set = CustomDataset(
            os.path.join(self.datasets_dir, 'processed_dev'),
            num_channels=self.params.num_channels,
            num_patches=self.params.num_patches,
            patch_size=self.params.patch_size,
            task_type=self.params.task_type,
        )
        test_set = CustomDataset(
            os.path.join(self.datasets_dir, 'processed_eval'),
            num_channels=self.params.num_channels,
            num_patches=self.params.num_patches,
            patch_size=self.params.patch_size,
            task_type=self.params.task_type,
        )

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))

        return {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
        }
