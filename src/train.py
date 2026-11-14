# -*- coding: utf-8 -*-
"""训练入口：一行命令跑一组实验。

示例：
    python src/train.py --cell RNN  --epochs 10 --lr 0.2
    python src/train.py --cell GRU  --epochs 10 --lr 0.2
    python src/train.py --cell RNN  --cell2 GRU --combine
    python src/train.py --cell MANUAL

每次训练结束会：
    reports/ckpt_<tag>.pt          权重 + 配置 + 类别顺序
    image/curves_<tag>.png         训练 loss + 验证准确率
    report/classification_report_<tag>.txt
    image/confusion_matrix_<tag>.png
    report/experiments.csv         追加一行实验记录
"""
import argparse
import csv
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CharNoiseWrapper, NamesDataset, stratified_split  # noqa: E402
from src.evaluate import evaluate_model                  # noqa: E402
from src.metrics import predict                          # noqa: E402
from src.model import CharRNN                            # noqa: E402
from src.vocab import N_LETTERS                          # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_set, batch_size, criterion, optimizer, device):
    model.train()
    order = list(range(len(train_set)))
    random.shuffle(order)
    batches = np.array_split(order, max(1, len(order) // batch_size))

    current_loss = 0.0
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
    return current_loss / len(batches)


def plot_curves(losses, val_accs, path):
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def append_experiment(path, row):
    """同一个 tag 重跑时覆盖旧行，而不是追加（避免表里出现重复配置）。"""
    rows, old_fields = [], []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if r.get("tag") != row["tag"]]
            old_fields = list(reader.fieldnames or [])

    # 表里可能已经有额外列（例如 carry_best.py 加的 note），这里取并集
    fields = list(row.keys())
    for f in old_fields:
        if f not in fields:
            fields.append(f)

    rows.append({k: str(v) for k, v in row.items()})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="RNN",
                        help="RNN / GRU / LSTM / MANUAL")
    parser.add_argument("--cell2", default="GRU", help="combine 时的第二层")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--round", type=int, default=1,
                        help="第几轮实验，结果写入 experiments_round<N>.csv")
    parser.add_argument("--scheduler", default="none",
                        choices=["none", "step", "lambda"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--aug", type=float, default=0.0,
                        help="字符级增强概率，0 表示关闭")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data", "names"))
    parser.add_argument("--no-eval", action="store_true", help="不跑测试集评估")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_dir = os.path.join(HERE, "report")
    img_dir = os.path.join(HERE, "image")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    tag = ("%s_lr%g_ep%d_bs%d_h%d_mid%d_l%d"
           % (args.cell, args.lr, args.epochs, args.batch_size,
              args.hidden, args.mid, args.layers))
    if args.combine:
        tag += "_combine%s" % args.cell2
    if args.dropout > 0:
        tag += "_do%g" % args.dropout
    if args.attention:
        tag += "_attn"
    if args.scheduler != "none":
        tag += "_sch%s" % args.scheduler
    if args.aug > 0:
        tag += "_aug%g" % args.aug

    print("config:", vars(args))
    print("tag:", tag)

    dataset = NamesDataset(args.data_dir)
    train_set, val_set, test_set = stratified_split(dataset, args.seed)
    if args.aug > 0:
        train_set = CharNoiseWrapper(train_set, p_drop=args.aug, p_swap=args.aug)
    print("split: train=%d val=%d test=%d"
          % (len(train_set), len(val_set), len(test_set)))

    model = CharRNN(
        input_size=N_LETTERS, hidden_size=args.hidden,
        output_size=len(dataset.labels_uniq),
        cell=args.cell, cell2=args.cell2, combine=args.combine,
        num_layers=args.layers, mid_size=args.mid,
        dropout=args.dropout, attention=args.attention).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters: {:,}".format(n_params))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    scheduler = None
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == "lambda":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: 0.95 ** e)

    losses, val_accs = [], []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_set, args.batch_size,
                               criterion, optimizer, device)
        y_true, y_pred = predict(model, val_set, device)
        val_acc = float((y_true == y_pred).mean())
        losses.append(loss)
        val_accs.append(val_acc)
        if scheduler is not None:
            scheduler.step()
        print("epoch %2d/%d  train_loss=%.4f  val_acc=%.4f"
              % (epoch, args.epochs, loss, val_acc))
    train_time = time.perf_counter() - t0
    print("training took %.1f s" % train_time)

    plot_curves(losses, val_accs, os.path.join(img_dir, "curves_%s.png" % tag))

    ckpt_path = os.path.join(report_dir, "ckpt_%s.pt" % tag)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": dataset.labels_uniq,
        "config": {"cell": args.cell, "cell2": args.cell2,
                   "combine": args.combine, "n_letters": N_LETTERS,
                   "hidden": args.hidden, "mid_size": args.mid,
                   "layers": args.layers, "lr": args.lr,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "seed": args.seed, "dropout": args.dropout,
                   "attention": args.attention, "aug": args.aug},
        "losses": losses, "val_accs": val_accs,
    }, ckpt_path)
    print("saved ->", ckpt_path)

    test_acc, test_f1 = float("nan"), float("nan")
    if not args.no_eval:
        result = evaluate_model(model, test_set, dataset.labels_uniq, tag,
                                report_dir, img_dir, device)
        test_acc, test_f1 = result["acc"], result["macro_f1"]

    exp_path = os.path.join(report_dir, "experiments_round%d.csv" % args.round)
    append_experiment(exp_path, {
        "round": args.round, "tag": tag, "cell": args.cell, "cell2": args.cell2,
        "combine": args.combine, "hidden": args.hidden, "mid_size": args.mid,
        "layers": args.layers, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "seed": args.seed,
        "dropout": args.dropout, "attention": args.attention, "aug": args.aug,
        "scheduler": args.scheduler,
        "note": "",
        "params": n_params, "train_time_s": round(train_time, 1),
        "best_val_acc": round(max(val_accs), 4),
        "test_acc": test_acc, "test_macro_f1": test_f1,
    })
    print("saved ->", exp_path)


if __name__ == "__main__":
    main()
