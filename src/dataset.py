# -*- coding: utf-8 -*-
"""数据集与分层划分。

对应 base 的第 87-130 行。改动点：
    [改] 读完一行后先 unicode_to_ascii（清洗 + 小写），再建张量
    [改] labels_uniq 用 sorted()，保证每次运行的类别顺序一致
    [改] 用 train_test_split(stratify=...) 做 70/15/15 分层划分，
         替代原来的 random_split（原版只有 train/test 且不保证类别比例）
"""
import glob
import os
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from .vocab import line_to_tensor, unicode_to_ascii


class NamesDataset(Dataset):
    """读取 data_dir 下所有 *.txt，每个文件是一个国籍类别。"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.strftime("%Y-%m-%d %H:%M:%S")

        self.data = []            # 清洗后的姓名（小写）
        self.tensors = []         # one-hot 张量
        self.labels = []          # 类别名
        self.label_tensors = []   # 类别下标张量

        text_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not text_files:
            raise FileNotFoundError(
                "在 %s 下没有找到 *.txt，请确认 DATA_DIR 是否正确" % data_dir)

        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            with open(filename, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    name = unicode_to_ascii(raw)          # [改] 清洗 + 小写
                    if not name:
                        continue
                    self.data.append(name)
                    self.tensors.append(line_to_tensor(name))
                    self.labels.append(label)

        # [改] sorted 保证类别下标与类别名的映射每次运行都相同
        self.labels_uniq = sorted(set(self.labels))
        index = {c: i for i, c in enumerate(self.labels_uniq)}
        self.label_tensors = [
            torch.tensor([index[l]], dtype=torch.long) for l in self.labels
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.label_tensors[idx], self.tensors[idx],
                self.labels[idx], self.data[idx])


def stratified_split(dataset, seed=2026, val_ratio=0.15, test_ratio=0.15):
    """按类别分层划分训练/验证/测试集（默认 70/15/15）。

    返回三个 torch.utils.data.Subset，用法与普通 Dataset 相同。
    """
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels[rest_idx],
        random_state=seed,
    )
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


class CharNoiseWrapper(Dataset):
    """字符级数据增强：随机删一个字符 / 交换相邻字符。

    只应该套在训练集上。姓名平均只有 6 个字符，所以概率要给得很小
    （p=0.05~0.1），否则会把名字改得认不出来。
    """

    def __init__(self, subset, p_drop=0.05, p_swap=0.05):
        self.subset = subset
        self.p_drop = p_drop
        self.p_swap = p_swap

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        label_t, _, label, name = self.subset[idx]
        if len(name) > 3:
            chars = list(name)
            if random.random() < self.p_drop:
                del chars[random.randrange(len(chars))]
            if random.random() < self.p_swap and len(chars) > 1:
                j = random.randrange(len(chars) - 1)
                chars[j], chars[j + 1] = chars[j + 1], chars[j]
            name = "".join(chars)
        return label_t, line_to_tensor(name), label, name
