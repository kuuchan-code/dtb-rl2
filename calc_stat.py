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
parser.add_argument(
    "-n", "--new", help="新しいデータのみ (データ数を与える)", type=int
)

args = parser.parse_args()

r = pyper.R(use_pandas=True)
for i in range(10):
    r("")


def main(fnamer):
    # print(fnamer)
    df = pd.read_csv(fnamer)
    r.assign("df", df)
    # print(r('df["animals"]'))
    moji = r("summary(df)")
    # t検定
    moji += r('t.test(df["animals"])')
    moji += r('t.test(df["height"])')
    # ピアソンの積率相関係数
    moji += r('cor.test(df[,1], df[,2])')
    print(moji)
    fnamew = f"statistics/{os.path.basename(fnamer).split('.', 1)[0]}.txt"
    with open(fnamew, "w") as f:
        print(moji, file=f)
    # print(r('df[,1]'))


if __name__ == "__main__":
    main(args.file)
