#!/usr/bin/env python3
"""
統計値計算
"""
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


class RangeCheck(object):
    def __init__(self, low_limit=None, high_limit=None, vtype="integer"):
        self.min = low_limit
        self.max = high_limit
        self.type = vtype

    def __contains__(self, val):
        ret = True
        if self.min is not None:
            ret = ret and (val >= self.min)
        if self.max is not None:
            ret = ret and (val <= self.max)
        return ret

    def __iter__(self):
        low = self.min
        if low is None:
            low = "-inf"
        high = self.max
        if high is None:
            high = "+inf"
        l1 = self.type
        l2 = f" {low} <= x <= {high}"
        return iter((l1, l2))


parser = argparse.ArgumentParser(description="学習の推移を描画")
parser.add_argument("file", help="読み込むファイル")
parser.add_argument("N", help="行動パターン数", type=int, choices=RangeCheck(low_limit=1), default=50
                    )
parser.add_argument("-s", "--step", help="特定ステップの行動だけ抽出",
                    type=int, choices=RangeCheck(low_limit=0))


args = parser.parse_args()


def main(fnamer, n, step):
    print(fnamer)
    df = pd.read_csv(fnamer)
    if step is not None:
        df = df[df["step"] == step]
    fig = plt.figure(figsize=(8, 5))
    # print(set(df["action"]))
    ax = fig.add_subplot(111)
    ax.hist(df["action"], bins=n, range=(-0.5, n - 0.5))
    ax.set_xlabel("action")
    # ax.set_ylabel("animals, height")
    ax.grid()
    # print(y1.shape)
    # print(y1_conv.shape)
    plt.show()


if __name__ == "__main__":
    main(args.file, args.N, args.step)
