# -*- coding: utf-8 -*-
"""字符表与「姓名 -> one-hot 张量」的转换。

对应 base 的第 56-84 行。改动点：
    [改] 字符表由 ascii_letters 改为 ascii_lowercase（配合统一小写）
    [改] unicode_to_ascii 末尾统一 .lower()
"""
import string
import unicodedata

import torch

# [改] 只用小写字母：26 + 空格/标点 5 + "_" = 32
ALLOWED_CHARACTERS = string.ascii_lowercase + " .,;'" + "_"
N_LETTERS = len(ALLOWED_CHARACTERS)


def unicode_to_ascii(s):
    """Unicode -> ASCII：去掉重音符号，只保留字符表内的字符，最后统一小写。

    'Ślusàrski' -> 'slusarski'
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALLOWED_CHARACTERS
    ).lower()                                   # [改] 原来没有 .lower()


def letter_to_index(letter):
    """字母 -> 索引；不在字符表里的字符用 "_" 的下标代替（OOV）。"""
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
    """模型输出 -> (类别名, 类别下标)。"""
    top_i = output.topk(1)[1][0].item()
    return classes[top_i], top_i
