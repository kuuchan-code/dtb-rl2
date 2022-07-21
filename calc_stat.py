#!/usr/bin/env python3
"""
統計値計算
"""
from __future__ import annotations
import pyper
import pandas as pd
import argparse
import os


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


parser = argparse.ArgumentParser(description="エピソード終了時の動物数, 高さの統計量計算")

parser.add_argument("file", help="読み込むファイル")
parser.add_argument(
    "-n", "--new", help="新しいデータのみ (データ数を与える)", type=int, choices=RangeCheck(low_limit=10)
)


args = parser.parse_args()

r = pyper.R(use_pandas=True)
for i in range(10):
    r("")


def main(fnamer: str, new_data_num: int | None):
    df = pd.read_csv(fnamer)
    b_name = os.path.splitext(os.path.basename(fnamer))[0]
    if new_data_num is None:
        fnamew = f"statistics/{b_name}.txt"
    else:
        df = df[-new_data_num:]
        fnamew = f"statistics/{b_name}_new{new_data_num}.txt"
    # print(df)
    r.assign("df", df)
    # print(r('df["animals"]'))
    moji = r("summary(df)")
    # t検定
    moji += r('t.test(df["animals"])')
    moji += r('t.test(df["height"])')
    # ピアソンの積率相関係数
    moji += r('cor.test(df[,1], df[,2])')
    print(moji)
    with open(fnamew, "w") as f:
        print(moji, file=f)
    # print(r('df[,1]'))


if __name__ == "__main__":
    main(args.file, args.new)
