import os
from pathlib import Path

import numpy as np
import joblib

from torch.utils.data import Dataset

from utils.utils import cal_lps

class TestDataset(Dataset):
    """
    定义测试集
    """

    def __init__(self,
                 mixture_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/test/mixture.pkl",
                 clean_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/test/clean.pkl",
                 limit=None,
                 offset=0,
                 apply_normalization=0
                 ):
        """
        构建测试数据集
        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)

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
            mixture_dataset_keys = mixture_dataset_keys[offset: offset + self.length] # offset + self.length 即便大于原数据总长度也不会报错
            self.length = len(mixture_dataset_keys) # 处理 offset + self.length 超过原数据集长度的问题

        norm_meta = {
            "type": apply_normalization,
            "eigenvalues": {}
        }
        if apply_normalization == 1:
            print(f"使用基于全部数据的 max 与 min 进行 [-1, 1] 归一化")
            print("正在计算测试集中全局规范化所需要的特征值...")
            max_total = -float("inf")
            min_total = float("inf")

            for key in mixture_dataset_keys:
                if np.max(cal_lps(mixture_dataset[key])) > max_total:
                    max_total = np.max(cal_lps(mixture_dataset[key]))
                if np.min(cal_lps(mixture_dataset[key])) < min_total:
                    min_total = np.min(cal_lps(mixture_dataset[key]))

            print(f"mixture 数据集的 max 特征值为 {max_total}, min 特征值为 {min_total}")
            norm_meta["eigenvalues"]["max"] = max_total
            norm_meta["eigenvalues"]["min"] = min_total

        self.mixture_dataset = mixture_dataset
        self.clean_dataset = clean_dataset
        self.mixture_dataset_keys = mixture_dataset_keys
        self.norm_meta = norm_meta

        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")
        print(f"len(self.length) is {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_name = self.mixture_dataset_keys[item]
        num = mixture_name.split("_")[0]
        mixture = self.mixture_dataset[mixture_name]
        clean = self.clean_dataset[num]

        assert mixture.shape == clean.shape
        return mixture.reshape(-1), clean.reshape(-1), mixture_name, self.norm_meta