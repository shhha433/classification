# -*- coding: utf-8 -*-
"""在验证集/测试集上评估已训练的模型。

用法：
    python src/evaluate.py --ckpt report/ckpt_RNN_lr0.2_ep10_bs64_h128_mid0_l1.pt

输出：
    report/classification_report_<tag>.txt
    image/confusion_matrix_<tag>.png
    report/eval_<tag>.json
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import NamesDataset, stratified_split   # noqa: E402
from src.metrics import metrics, predict                 # noqa: E402
from src.model import CharRNN                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(config, output_size):
    return CharRNN(
        input_size=config["n_letters"],
        hidden_size=config["hidden"],
        output_size=output_size,
        cell=config["cell"],
        cell2=config["cell2"],
        combine=config["combine"],
        num_layers=config["layers"],
        mid_size=config["mid_size"],
        dropout=config.get("dropout", 0.0),
        attention=config.get("attention", False),
    )


def evaluate_model(model, dataset, classes, tag,
                   report_dir, img_dir, device):
    """算指标、写报告、画混淆矩阵，返回 dict。"""
    y_true, y_pred = predict(model, dataset, device)
    acc, macro_f1 = metrics(y_true, y_pred)
    # 标签名要按“下标 -> 类别名”的顺序给，classes 必须与训练时一致
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4, zero_division=0)

    print("accuracy = %.4f   macro_f1 = %.4f" % (acc, macro_f1))
    print(report)

    txt_path = os.path.join(report_dir, "classification_report_%s.txt" % tag)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("accuracy = %.4f\nmacro_f1 = %.4f\n\n%s" % (acc, macro_f1, report))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_norm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "confusion_matrix_%s.png" % tag), dpi=150)
    plt.close(fig)

    return {"acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "report_path": txt_path}


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    classes = ckpt["classes"]
    model = build_model(config, len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="训练保存的 .pt")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data", "names"))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    report_dir = os.path.join(HERE, "report")
    img_dir = os.path.join(HERE, "image")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, classes = load_checkpoint(args.ckpt, device)
    dataset = NamesDataset(args.data_dir)
    if dataset.labels_uniq != classes:
        print("警告：数据集的类别顺序与 checkpoint 不一致，已使用 checkpoint 的顺序")

    train_set, val_set, test_set = stratified_split(dataset, config["seed"])
    target = test_set if args.split == "test" else val_set

    tag = os.path.splitext(os.path.basename(args.ckpt))[0].replace("ckpt_", "")
    result = evaluate_model(model, target, classes, tag,
                            report_dir, img_dir, device)
    result.update({"split": args.split, "n_samples": len(target),
                   "tag": tag, "config": config})
    with open(os.path.join(report_dir, "eval_%s.json" % tag), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("saved -> report/eval_%s.json" % tag)


if __name__ == "__main__":
    main()
