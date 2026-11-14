# -*- coding: utf-8 -*-
"""预测与指标：训练时算验证准确率，评估时算测试集报告。"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def predict(model, dataset, device=None):
    """对整个数据集逐条预测，返回 (真实标签下标, 预测标签下标)。"""
    device = device or torch.device("cpu")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            label_t, text_t, _, _ = dataset[i]
            output = model(text_t.to(device))
            y_true.append(int(label_t.item()))
            y_pred.append(int(output.argmax(1).item()))
    return np.array(y_true), np.array(y_pred)


def metrics(y_true, y_pred):
    """返回 (accuracy, macro F1)。"""
    return (float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
