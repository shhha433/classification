# 基于 RNN 的姓名国籍分类

字符级循环神经网络，输入一个人名（如 `Zhang`、`Al-Sayed`），判断它属于哪个国家/语言。

## 目录结构

```
RNN/
├── base.py                  # 单文件完整版（训练时只改顶部开关，最方便调参）
├── src/                     # 拆分版
│   ├── vocab.py             # 字符表、姓名 -> one-hot 张量
│   ├── dataset.py           # 读取数据、分层划分
│   ├── model.py             # 手写 RNN 单元 + CharRNN
│   ├── metrics.py           # 预测与指标
│   ├── data_stats.py        # 类别分布 / 姓名长度统计
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 测试集评估入口
│   └── carry_best.py        # 把上一轮冠军带到下一轮表首行
├── data/names/              # 原始数据：18 个 txt，每行一个姓名
├── image/                   # 曲线、混淆矩阵、统计图
├── report/                  # 分类报告、逐轮实验表、模型权重
├── run_all.sh               # 一次跑完三轮实验
├── README.md
```

## 环境

Python 3.10+，CPU 即可运行。

```bash
pip install torch numpy matplotlib scikit-learn
```

本机验证过的版本：Python 3.14.7 / torch 2.14.0 / scikit-learn 1.9.1。

## 数据

`data/names/*.txt`，共 18 个类别、20,074 条姓名。预处理规则：

1. Unicode 归一化（NFD）并去掉重音符号，`Ślusàrski → slusarski`
2. 统一转小写
3. 只保留字符表 `a-z`、空格、`. , ; '`、`_`（`_` 表示 OOV）
4. 字符表大小 32，姓名转成 `(len, 1, 32)` 的 one-hot 张量
5. 按类别分层划分 70% / 15% / 15%

## 复现

### 1. 数据统计（阶段一）

```bash
python src/data_stats.py
```

生成 `image/class_distribution.png`、`image/name_length.png`、`report/class_counts.csv`。

### 2. 训练单组实验

```bash
# 基线
python src/train.py --cell RNN --epochs 10 --lr 0.2

# 手写 RNN 单元（作业阶段二要求 a）
python src/train.py --cell MANUAL

# GRU / LSTM
python src/train.py --cell GRU
python src/train.py --cell LSTM

# 改隐藏层大小
python src/train.py --cell RNN --hidden 64
python src/train.py --cell RNN --hidden 256

# 加一层全连接
python src/train.py --cell RNN --mid 64

# 两个 RNN 串联
python src/train.py --cell RNN --cell2 GRU --combine

# 动态学习率（每 5 轮 x0.5 或 0.95^epoch）
python src/train.py --cell RNN --scheduler step
python src/train.py --cell RNN --scheduler lambda

# dropout / 注意力池化 / 字符级数据增强
python src/train.py --cell RNN --dropout 0.2
python src/train.py --cell RNN --attention
python src/train.py --cell RNN --aug 0.05
```

训练结束会自动跑测试集评估，并写入 `report/experiments_round<轮次>.csv`。
默认第 1 轮；用 `--round 2` 指定其他轮次。同一个 tag 重跑会覆盖旧行，不会重复。

### 3. 单独评估已有模型

```bash
python src/evaluate.py --ckpt report/ckpt_RNN_lr0.2_ep10_bs64_h128_mid0_l1.pt
```

### 4. 一次跑完实验矩阵

```bash
bash run_all.sh        # 三轮全跑
bash run_all.sh 1      # 只跑第一轮
bash run_all.sh 2      # 只跑第二轮
bash run_all.sh 3      # 只跑第三轮
```

## 输出说明

| 文件 | 内容 |
|---|---|
| `report/ckpt_<tag>.pt` | 权重 + 配置 + 类别顺序 + 每轮 loss/val_acc |
| `report/experiments_round<N>.csv` | 第 N 轮实验的配置、指标、参数量与耗时 |
| `report/classification_report_<tag>.txt` | 每类 precision / recall / F1 |
| `image/curves_<tag>.png` | 训练 loss + 验证准确率曲线 |
| `image/confusion_matrix_<tag>.png` | 归一化混淆矩阵 |

## 复现性

所有随机源（`random`、`numpy`、`torch`、`torch.cuda`）统一使用种子 `2026`；
数据划分、模型初始化、batch 打乱都由该种子控制，同一配置重复运行结果一致。



