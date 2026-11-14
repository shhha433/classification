#!/usr/bin/env bash
# 一次跑完三轮实验（可用参数只跑其中一轮）
# 用法：
#   bash run_all.sh          # 三轮全跑
#   bash run_all.sh 1        # 只跑第一轮（单变量消融）
#   bash run_all.sh 2        # 只跑第二轮（GRU 调参）
#   bash run_all.sh 3        # 只跑第三轮（冠军复核）
set -e

cd "$(dirname "$0")"
PY="${PYTHON:-python}"
MODE="${1:-all}"

round1() {
  echo "========== 第一轮：单变量消融 =========="
  $PY src/data_stats.py

  $PY src/train.py --round 1 --cell RNN    --epochs 10 --lr 0.2 --batch-size 64
  $PY src/train.py --round 1 --cell GRU    --epochs 10 --lr 0.2 --batch-size 64
  $PY src/train.py --round 1 --cell LSTM   --epochs 10 --lr 0.2 --batch-size 64
  $PY src/train.py --round 1 --cell MANUAL --epochs 10 --lr 0.2 --batch-size 64

  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --hidden 64
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --hidden 256
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --mid 64
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --layers 2
  $PY src/train.py --round 1 --cell RNN --cell2 GRU --combine --epochs 10 --lr 0.2 --batch-size 64

  # 三变量调参组合，报告中标为“调参组合”，不参与单变量归因
  $PY src/train.py --round 1 --cell RNN --epochs 30 --lr 0.1 --batch-size 128

  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --scheduler step
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --scheduler lambda
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --dropout 0.2
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --attention
  $PY src/train.py --round 1 --cell RNN --epochs 10 --lr 0.2 --batch-size 64 --aug 0.05
}

round2() {
  echo "========== 第二轮：以 GRU 为基础调参 =========="
  $PY src/train.py --round 2 --cell GRU --epochs 10 --lr 0.1  --batch-size 64
  $PY src/train.py --round 2 --cell GRU --epochs 10 --lr 0.05 --batch-size 64
  $PY src/train.py --round 2 --cell GRU --epochs 20 --lr 0.2  --batch-size 64
  $PY src/train.py --round 2 --cell GRU --epochs 30 --lr 0.2  --batch-size 64
  $PY src/train.py --round 2 --cell GRU --epochs 10 --lr 0.2  --batch-size 128
  $PY src/train.py --round 2 --cell GRU --epochs 10 --lr 0.2  --batch-size 32
  $PY src/train.py --round 2 --cell GRU --epochs 30 --lr 0.2  --batch-size 64 --scheduler step
}

round3() {
  echo "========== 第三轮：冠军复核与结构扩展 =========="
  $PY src/train.py --round 3 --cell GRU --epochs 30 --lr 0.2 --batch-size 64 --scheduler step
  $PY src/train.py --round 3 --cell GRU --epochs 30 --lr 0.2 --batch-size 64 --layers 2
}

case "$MODE" in
  1) round1 ;;
  2) round2 ;;
  3) round3 ;;
  all) round1; round2; round3 ;;
  *) echo "用法: bash run_all.sh [1|2|3|all]"; exit 1 ;;
esac

echo
echo "完成。结果："
echo "  report/experiments_round1.csv"
echo "  report/experiments_round2.csv"
echo "  report/experiments_round3.csv"
echo "  report/classification_report_*.txt"
echo "  image/curves_*.png  image/confusion_matrix_*.png"
