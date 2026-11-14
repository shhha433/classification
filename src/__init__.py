# -*- coding: utf-8 -*-
"""姓名国籍分类项目（拆分自 base，便于提交与复现）

模块职责：
    vocab.py      字符表、字符串 -> 张量
    dataset.py    读取数据、分层划分
    model.py      手写 RNN 单元 + CharRNN
    metrics.py    预测与准确率
    data_stats.py 数据统计图
    train.py      训练入口
    evaluate.py   测试集评估入口
"""
