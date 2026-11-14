# -*- coding: utf-8 -*-
"""把上一轮验证集最好的配置抄到下一轮表格的第一行，方便对比。

用法：
    python src/carry_best.py --from-round 1 --to-round 2
    python src/carry_best.py --from-round 2 --to-round 3
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-round", type=int, required=True)
    parser.add_argument("--to-round", type=int, required=True)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "report"))
    args = parser.parse_args()

    src = os.path.join(args.out_dir, "experiments_round%d.csv" % args.from_round)
    dst = os.path.join(args.out_dir, "experiments_round%d.csv" % args.to_round)

    src_rows, fields = load(src)
    if not src_rows:
        sys.exit("没有找到 %s 或表是空的" % src)

    # 用验证集选冠军；验证集相同则看 macro F1
    best = max(src_rows, key=lambda r: (float(r["best_val_acc"]),
                                        float(r.get("test_macro_f1") or 0)))
    dst_rows, dst_fields = load(dst)

    for f in dst_fields:
        if f not in fields:
            fields.append(f)
    if "note" not in fields:
        fields.append("note")

    dst_rows = [r for r in dst_rows if r.get("tag") != best["tag"]]

    carry = dict(best)
    carry["round"] = args.to_round
    carry["note"] = "carry-over from round %d" % args.from_round

    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerow(carry)
        writer.writerows(dst_rows)

    print("round%d 的冠军已放到 round%d 第一行：" % (args.from_round, args.to_round))
    print("  %-46s val %.4f  test %.4f"
          % (best["tag"], float(best["best_val_acc"]), float(best["test_acc"])))


if __name__ == "__main__":
    main()
