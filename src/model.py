# -*- coding: utf-8 -*-
"""模型定义：手写 RNN 单元 + 内置 RNN/GRU/LSTM。

对应 base 的第 132-173 行。改动点：
    [改] 补全 ManualRNNCell（作业阶段二要求 (a)：两个线性层 + tanh + 隐状态传递）
         原 base 里写了 i2o 却没用、还引用了不存在的 self.h2h
    [改] 用 make_cell 统一构造，支持 "RNN"/"GRU"/"LSTM"/"MANUAL"
    [改] 手写单元只返回最终隐状态，这里用 _run 把两种接口统一成
         (序列输出, 最终隐状态)，forward 不再依赖 out[-1] 的偶然巧合
    [改] 组合网络时第二层接的是第一层的整段序列输出（与 base 的写法一致）
"""
import torch
import torch.nn as nn


class ManualRNNCell(nn.Module):
    """手写 RNN：h_t = tanh(W_ih * x_t + W_hh * h_{t-1}）

    没有门控，就是最原始的 RNN。作业要求 (a) 用它和 nn.RNN 对比。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq, h=None):
        """seq: (seq_len, batch, input_size) -> 最终隐状态 (batch, hidden)"""
        if h is None:
            h = torch.zeros(seq.size(1), self.hidden_size, device=seq.device)
        for t in range(seq.size(0)):
            h = torch.tanh(self.i2h(seq[t]) + self.h2h(h))
        return h


def make_cell(name, input_size, hidden_size, num_layers=1):
    """按名字造一个循环层。MANUAL 是手写单元，其余交给 torch.nn。"""
    name = name.upper()
    if name == "MANUAL":
        return ManualRNNCell(input_size, hidden_size)
    return getattr(nn, name)(input_size, hidden_size, num_layers=num_layers)


def run_layer(layer, seq):
    """统一内置层与手写单元的接口。

    返回 (序列输出, 最终隐状态)：
        内置层：out 是 (seq_len, batch, hidden)，h 是最后一步
        手写单元：只有最终隐状态，补一个长度为 1 的序列方便继续往下接
    """
    if isinstance(layer, ManualRNNCell):
        h = layer(seq)
        return h.unsqueeze(0), h
    out, _ = layer(seq)          # LSTM 这里返回 (h, c)，用 _ 接住即可
    return out, out[-1]


class CharRNN(nn.Module):
    """字符级分类模型。

    cell / cell2 : "RNN" / "GRU" / "LSTM" / "MANUAL"
    combine      : True 时把 cell 和 cell2 串起来（更高层网络）
    num_layers   : 内置循环层的堆叠数
    mid_size     : >0 时在分类头前插一层全连接 + ReLU
    """

    def __init__(self, input_size, hidden_size, output_size,
                 cell="RNN", cell2="GRU", combine=False,
                 num_layers=1, mid_size=0,
                 dropout=0.0, attention=False):
        super().__init__()
        self.combine = combine
        self.attention = attention
        self.cell1 = make_cell(cell, input_size, hidden_size, num_layers)
        if combine:
            # [改] 第二层的输入维度必须是第一层的输出维度 hidden_size
            self.cell2 = make_cell(cell2, hidden_size, hidden_size, num_layers)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if attention:
            self.attn = nn.Linear(hidden_size, 1)

        if mid_size > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid_size),
                nn.ReLU(),
                nn.Linear(mid_size, output_size),
            )
        else:
            self.head = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq):
        seq_out, h = run_layer(self.cell1, seq)
        if self.combine:
            seq_out, h = run_layer(self.cell2, seq_out)

        if self.attention:
            # 注意力池化：给每个时间步打分，再按权重加权求和，
            # 代替"只看最后一个时间步"的做法
            scores = self.attn(seq_out).squeeze(-1)          # (seq_len, batch)
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            h = (seq_out * weights).sum(dim=0)               # (batch, hidden)

        return self.log_softmax(self.head(self.drop(h)))
