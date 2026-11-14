# -*- coding: utf-8 -*-

import os
import random
import string
import time
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

# ==========================================================================
# 配置
# ==========================================================================
SEED = 2026

CELL = "LSTM"          # "RNN" / "GRU" / "LSTM" / "MANUAL"（手写单元）
CELL2 = "GRU"         # combine=True 时第二层用什么
combine = False       # True = 两个循环层串联

LR = 0.2
epochs = 10
batch_size = 64
n_hidden = 128
mid_size = 0          # >0 时分类头前加一层 全连接 + ReLU
layers = 1            # 内置循环层的堆叠数

PRINT_EVERY = 1       # 每几个 epoch 打印一次进度

ROUND = 1             # 第几轮实验，结果写入 experiments_round<N>.csv
SCHEDULER = "none"    # "none" / "step"（每5轮x0.5） / "lambda"（0.95^epoch）
DROPOUT = 0.0         # >0 时在分类头前加 dropout
ATTENTION = False     # True = 注意力池化，替代只看最后一步
AUG = 0.0             # 字符级增强概率，0 表示关闭
# ==========================================================================


def set_seed(seed):                                            
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "names")
REPORT_DIR = os.path.join(HERE, "report")
IMG_DIR = os.path.join(HERE, "image")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

tag = ("%s_lr%g_ep%d_bs%d_h%d_mid%d_l%d"
       % (CELL, LR, epochs, batch_size, n_hidden, mid_size, layers))
if combine:
    tag += "_combine%s" % CELL2
if DROPOUT > 0:
    tag += "_do%g" % DROPOUT
if ATTENTION:
    tag += "_attn"
if SCHEDULER != "none":
    tag += "_sch%s" % SCHEDULER
if AUG > 0:
    tag += "_aug%g" % AUG

# ==========================================================================
# 字符表
# ==========================================================================
ALLOWED_CHARACTERS = string.ascii_lowercase + " .,;'" + "_"     
N_LETTERS = len(ALLOWED_CHARACTERS)


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALLOWED_CHARACTERS
    ).lower()                                                    

def letter_to_index(letter):
    if letter not in ALLOWED_CHARACTERS:
        return ALLOWED_CHARACTERS.find("_")
    return ALLOWED_CHARACTERS.find(letter)


def line_to_tensor(line):
    """姓名 -> (len, 1, n_letters) 的 one-hot 张量。"""
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(output, classes):
    top_i = output.topk(1)[1][0].item()
    return classes[top_i], top_i


# ==========================================================================
# 数据集与划分
# ==========================================================================
class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data, self.tensors, self.labels = [], [], []

        text_files = sorted(__import__("glob").glob(os.path.join(data_dir, "*.txt")))
        if not text_files:
            raise FileNotFoundError("在 %s 下没有找到 *.txt" % data_dir)

        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            with open(filename, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    name = unicode_to_ascii(raw)                 
                    if not name:
                        continue
                    self.data.append(name)
                    self.tensors.append(line_to_tensor(name))
                    self.labels.append(label)

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


def stratified_split(dataset, seed, val_ratio=0.15, test_ratio=0.15):
    """ 分层划分 70/15/15。"""
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)
    train_idx, rest_idx = train_test_split(
        indices, test_size=val_ratio + test_ratio,
        stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels[rest_idx], random_state=seed)
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


class CharNoiseWrapper(Dataset):
    """字符级数据增强：随机删字符 / 交换相邻字符，只套在训练集上。"""

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


# ==========================================================================
# 模型
# ==========================================================================
class ManualRNNCell(nn.Module):                                 
    """手写 RNN：h_t = tanh(W_ih x_t + W_hh h_{t-1})。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq):
        h = torch.zeros(seq.size(1), self.hidden_size, device=seq.device)
        for t in range(seq.size(0)):
            h = torch.tanh(self.i2h(seq[t]) + self.h2h(h))
        return h


def make_cell(name, input_size, hidden_size, num_layers):        
    name = name.upper()
    if name == "MANUAL":
        return ManualRNNCell(input_size, hidden_size)
    return getattr(nn, name)(input_size, hidden_size, num_layers=num_layers)


def run_layer(layer, seq):
    if isinstance(layer, ManualRNNCell):
        h = layer(seq)
        return h.unsqueeze(0), h
    out, _ = layer(seq)          # LSTM 返回 (h, c)，这里用 _ 接住
    return out, out[-1]


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 cell=CELL, cell2=CELL2, combine=combine,
                 num_layers=layers, mid_size=mid_size,
                 dropout=DROPOUT, attention=ATTENTION):
        super().__init__()
        self.combine = combine
        self.attention = attention
        self.cell1 = make_cell(cell, input_size, hidden_size, num_layers)
        if combine:
            self.cell2 = make_cell(cell2, hidden_size, hidden_size, num_layers)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if attention:
            self.attn = nn.Linear(hidden_size, 1)

        if mid_size > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid_size),
                nn.ReLU(),
                nn.Linear(mid_size, output_size))
        else:
            self.head = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq):
        seq_out, h = run_layer(self.cell1, seq)
        if self.combine:
            seq_out, h = run_layer(self.cell2, seq_out)
        if self.attention:
            scores = self.attn(seq_out).squeeze(-1)
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            h = (seq_out * weights).sum(dim=0)
        return self.log_softmax(self.head(self.drop(h)))


def predict(model, dataset, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            label_t, text_t, _, _ = dataset[i]
            output = model(text_t.to(device))
            y_true.append(int(label_t.item()))
            y_pred.append(int(output.argmax(1).item()))
    return np.array(y_true), np.array(y_pred)


def train(model, train_set, val_set, criterion, optimizer, device,
          scheduler=None):
    """训练 epochs 轮，每轮结束后在验证集上算准确率。"""
    losses, val_accs = [], []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        current_loss = 0.0
        order = list(range(len(train_set)))
        random.shuffle(order)
        batches = np.array_split(order, max(1, len(order) // batch_size))

        for batch in batches:
            model.zero_grad()
            batch_loss = 0
            for i in batch:
                label_t, text_t, _, _ = train_set[int(i)]
                output = model(text_t.to(device))
                batch_loss = batch_loss + criterion(output, label_t.to(device))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            current_loss += batch_loss.item() / len(batch)

        losses.append(current_loss / len(batches))

        y_true, y_pred = predict(model, val_set, device)
        val_acc = float((y_true == y_pred).mean())
        val_accs.append(val_acc)
        if scheduler is not None:
            scheduler.step()

        if epoch % PRINT_EVERY == 0:
            print("epoch %2d/%d  train_loss=%.4f  val_acc=%.4f"
                  % (epoch, epochs, losses[-1], val_acc))

    return losses, val_accs, time.perf_counter() - t0


def plot_curves(losses, val_accs):                               
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(range(1, len(losses) + 1), losses, "b-o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("average batch loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(val_accs) + 1), val_accs, "r-s", label="val accuracy")
    ax2.set_ylabel("validation accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "curves_%s.png" % tag), dpi=150)
    plt.close(fig)


def save_data_stats(dataset):                                    
    """类别分布 + 姓名长度分布。"""
    from collections import Counter

    counts = Counter(dataset.labels)
    names = sorted(counts)
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel("samples")
    ax.set_title("class distribution")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    lengths = [len(n) for n in dataset.data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color="darkorange")
    ax.set_xlabel("name length")
    ax.set_ylabel("count")
    ax.set_title("name length distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "name_length.png"), dpi=150)
    plt.close(fig)

    print("类别数 %d，样本数 %d，最短类别 %d，最长类别 %d"
          % (len(names), len(dataset), min(values), max(values)))
    print("姓名长度：平均 %.2f，最大 %d" % (np.mean(lengths), max(lengths)))


def evaluate(model, dataset, classes, device):                   # [修] 不再从外部传 F1
    y_true, y_pred = predict(model, dataset, device)
    acc = float((y_true == y_pred).mean())
    # [修] F1 必须用当前这批预测算，原来是拿验证集的 F1 写进测试报告
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    print("accuracy = %.4f   macro_f1 = %.4f" % (acc, macro_f1))
    print(report)

    with open(os.path.join(REPORT_DIR, "classification_report_%s.txt" % tag),
              "w", encoding="utf-8") as fh:
        fh.write("accuracy = %.4f\nmacro_f1 = %.4f\n\n%s" % (acc, macro_f1, report))

    confusion = np.zeros((len(classes), len(classes)))
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    row_sum = confusion.sum(axis=1, keepdims=True)
    confusion = confusion / np.maximum(row_sum, 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix_%s.png" % tag), dpi=150)
    plt.close(fig)
    return acc, macro_f1                                         # [修] 两个指标一起返回


def append_experiment(row):                                      
    """写入 report/experiments_round<N>.csv；同一 tag 重跑会覆盖旧行。"""
    import csv
    path = os.path.join(REPORT_DIR, "experiments_round%d.csv" % ROUND)
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("tag") != row["tag"]]
    rows.append({k: str(v) for k, v in row.items()})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row), restval="")
        writer.writeheader()
        writer.writerows(rows)


def main():                                                     
    print("config: cell=%s cell2=%s combine=%s hidden=%d mid=%d layers=%d "
          "lr=%g epochs=%d batch=%d seed=%d"
          % (CELL, CELL2, combine, n_hidden, mid_size, layers,
             LR, epochs, batch_size, SEED))

    alldata = NamesDataset(DATA_DIR)
    save_data_stats(alldata)
    train_set, val_set, test_set = stratified_split(alldata, SEED)   
    if AUG > 0:
        train_set = CharNoiseWrapper(train_set, p_drop=AUG, p_swap=AUG)
    print("split: train=%d val=%d test=%d"
          % (len(train_set), len(val_set), len(test_set)))

    model = CharRNN(N_LETTERS, n_hidden, len(alldata.labels_uniq),
                    dropout=DROPOUT, attention=ATTENTION).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters: {:,}".format(n_params))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    scheduler = None
    if SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif SCHEDULER == "lambda":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: 0.95 ** e)

    losses, val_accs, train_time = train(
        model, train_set, val_set, criterion, optimizer, device, scheduler)
    print("training took %.1f s" % train_time)

    plot_curves(losses, val_accs)                                

    # [修] 直接在测试集上评估，准确率和 F1 都来自同一批预测
    test_acc, macro_f1 = evaluate(model, test_set, alldata.labels_uniq, device)

    # 保存权重 + 配置 + 类别顺序（推理时要用）
    torch.save({
        "state_dict": model.state_dict(),
        "classes": alldata.labels_uniq,
        "config": {"cell": CELL, "cell2": CELL2, "combine": combine,
                   "n_letters": N_LETTERS, "hidden": n_hidden,
                   "mid_size": mid_size, "layers": layers,
                   "lr": LR, "epochs": epochs, "batch_size": batch_size,
                   "seed": SEED, "dropout": DROPOUT,
                   "attention": ATTENTION, "aug": AUG},
        "losses": losses,
        "val_accs": val_accs,
    }, os.path.join(REPORT_DIR, "ckpt_%s.pt" % tag))

    append_experiment({
        "round": ROUND, "tag": tag, "cell": CELL, "cell2": CELL2,
        "combine": combine,
        "hidden": n_hidden, "mid_size": mid_size, "layers": layers,
        "lr": LR, "epochs": epochs, "batch_size": batch_size, "seed": SEED,
        "dropout": DROPOUT, "attention": ATTENTION, "aug": AUG,
        "scheduler": SCHEDULER,
        "params": n_params, "train_time_s": round(train_time, 1),
        "best_val_acc": round(max(val_accs), 4),
        "test_acc": round(test_acc, 4), "test_macro_f1": round(macro_f1, 4),
    })
    print("saved -> report/ckpt_%s.pt" % tag)


if __name__ == "__main__":
    main()
