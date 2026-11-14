# -*- coding: utf-8 -*-
"""数据统计（阶段一交付物）：类别分布、姓名长度分布。

用法：
    cd "第二次作业 RNN"
    python src/data_stats.py

输出：
    image/class_distribution.png
    image/name_length.png
    report/class_counts.csv
"""
import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import NamesDataset                       # noqa: E402
from src.vocab import unicode_to_ascii                     # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data", "names"))
    args = parser.parse_args()

    report_dir = os.path.join(HERE, "report")
    img_dir = os.path.join(HERE, "image")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    dataset = NamesDataset(args.data_dir)
    counts = Counter(dataset.labels)
    names = sorted(counts)
    values = [counts[n] for n in names]
    lengths = np.array([len(n) for n in dataset.data])

    # ---- 表格 ----
    print("%-12s %6s" % ("class", "count"))
    for n, v in zip(names, values):
        print("%-12s %6d" % (n, v))
    print("\n类别数 %d  样本数 %d  最少 %d  最多 %d"
          % (len(names), len(dataset), min(values), max(values)))
    print("长度 均值 %.2f  中位数 %.1f  最大 %d"
          % (lengths.mean(), np.median(lengths), lengths.max()))

    with open(os.path.join(report_dir, "class_counts.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("class,count\n")
        for n, v in zip(names, values):
            fh.write("%s,%d\n" % (n, v))

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel("number of names")
    ax.set_title("class distribution")
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color="darkorange")
    ax.axvline(lengths.mean(), color="black", linestyle="--",
               label="mean = %.2f" % lengths.mean())
    ax.set_xlabel("name length (characters)")
    ax.set_ylabel("count")
    ax.set_title("name length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "name_length.png"), dpi=150)
    plt.close(fig)
    print("\n已保存: image/class_distribution.png, image/name_length.png, "
          "report/class_counts.csv")


if __name__ == "__main__":
    main()
