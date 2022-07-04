#!/usr/bin/env python3
"""
統計値計算
"""
import pyper
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="エピソード終了時の動物数, 高さの統計量計算")

parser.add_argument("file", help="読み込むファイル")

args = parser.parse_args()

r = pyper.R(use_pandas=True)
for i in range(10):
    r("")


def main(fnamer):
    # print(fnamer)
    df = pd.read_csv(fnamer)
    r.assign("df", df)
    # print(r('df["animals"]'))
    # t検定
    print(r("summary(df)"))
    print(r('t.test(df["animals"])'))
    print(r('t.test(df["height"])'))
    # ピアソンの積率相関係数
    print(r('cor.test(df[,1], df[,2])'))
    # print()
    fnamew = f"statistics/{os.path.basename(fnamer).split('.', 1)[0]}.txt"
    with open(fnamew, "w") as f:
        print(r("summary(df)"), file=f)
        print(r('t.test(df["animals"])'), file=f)
        print(r('t.test(df["height"])'), file=f)
        print(r('cor.test(df[,1], df[,2])'), file=f)
    # print(r('df[,1]'))


if __name__ == "__main__":
    main(args.file)
