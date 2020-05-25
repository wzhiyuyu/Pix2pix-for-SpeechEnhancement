import os

import numpy as np
import joblib
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0,
                 apply_normalization=0,
                 ):
        """

        Args:
            mixture_dataset:
            clean_dataset:
            limit:
            offset:
            apply_normalization(int): (0, 1, 2, 3)，0 表示不适用规范化，1表示语音整体使用规范化，2表示对单条语音使用规范化，3表示单帧规范化
        """
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)
        assert apply_normalization in (0, 1, 2, 3)

        print(f"Loading mixture dataset {mixture_dataset} ...")
        mixture_dataset: dict  = joblib.load(mixture_dataset)
        print(f"Loading clean dataset {clean_dataset} ...")
        clean_dataset: dict = joblib.load(clean_dataset)

        mixture_dataset_keys = list(mixture_dataset.keys())
        mixture_dataset_keys = sorted(mixture_dataset_keys)

        # Limit
        if limit and limit <= len(mixture_dataset_keys):
            self.length = limit
        else:
            self.length = len(mixture_dataset_keys)

        # Offset
        if offset:
            mixture_dataset_keys = mixture_dataset_keys[offset: offset + self.length]
            self.length = len(mixture_dataset_keys)

        self.mixture_dataset = mixture_dataset
        self.clean_dataset = clean_dataset
        self.mixture_dataset_keys = mixture_dataset_keys
        self.apply_normalization = apply_normalization

        if apply_normalization == 1:
            self.mixture_max_total = np.hstack(list(mixture_dataset.values())).max()
            self.mixture_min_total = np.hstack(list(mixture_dataset.values())).min()
            self.clean_max_total = np.hstack(list(clean_dataset.values())).max()
            self.clean_min_total = np.hstack(list(clean_dataset.values())).min()

            print(f"使用基于全部数据的 max 与 min 进行 [-1, 1] 归一化")
            print(f"mixture = ({self.mixture_max_total}, {self.mixture_min_total})")
            print(f"clean = ({self.clean_max_total}, {self.clean_min_total})")

        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")
        print(f"The len of fully dataset is {self.length}.")

    @staticmethod
    def norm_transform(max, min):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((max, ), (min, ))
        ])

    def __getitem__(self, item):
        mixture_name = self.mixture_dataset_keys[item]
        mixture = self.mixture_dataset[mixture_name]
        clean = self.clean_dataset[mixture_name.split("_")[0]]
        # Note: 去掉 librosa pad 而来的最后一帧
        mixture = mixture[:, :-1]
        clean = clean[:, :-1]
        assert mixture.shape == clean.shape

        # Aligned Sampling
        n_frames = 256
        frame_total = mixture.shape[1]
        start = np.random.randint(frame_total - n_frames + 1)
        end = start + n_frames
        mixture = mixture[:, start:end]
        clean = clean[:, start:end]
        # Note: dim 0: 257 => 256，2 的次方可能对于计算 gpu 计算更加友好
        mixture = mixture[:-1, :]
        clean = clean[:-1, :]

        assert mixture.shape == clean.shape == (256, 256)

        """
        一定要在添加新维度前进行规范化
        e.g.
            a = np.array([[[2,3,4],[1,2,3]]])
            a.shape
            (1, 2, 3)
            max = a.max(axis=0)
            array([[2, 3, 4],
                   [1, 2, 3]])
            max.shape
            (2, 3)
        """
        norm_meta = {} # apply_normalization == 0 时，info 为空
        if self.apply_normalization in (1,2,3):
            if self.apply_normalization == 1:
                mixture_max= self.mixture_max_total
                mixture_min  = self.mixture_min_total
                clean_max = self.clean_max_total
                clean_min = self.clean_min_total
            elif self.apply_normalization == 2:
                mixture_max = np.max(mixture)
                mixture_min = np.min(mixture)
                clean_max = np.max(clean)
                clean_min = np.min(clean)
            else:
                mixture_max = np.max(mixture, axis=0)
                mixture_min = np.min(mixture, axis=0)
                clean_max = np.max(clean, axis=0)
                clean_min = np.min(clean, axis=0)

            mixture = 2 * (mixture - mixture_min) / (mixture_max - mixture_min) - 1
            clean = 2 * (clean - clean_min) / (clean_max - clean_min) - 1

            norm_meta["mixture_max"] = mixture_max
            norm_meta["mixture_min"] = mixture_min
            norm_meta["clean_max"] = clean_max
            norm_meta["clean_min"] = clean_min


        return mixture.reshape(1, 256, 256), clean.reshape(1, 256, 256), mixture_name

    def __len__(self):
        return self.length
