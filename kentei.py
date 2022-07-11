#!/usr/bin/env python3
"""
統計値計算
"""
from __future__ import annotations
import pyper
import pandas as pd
import os


r = pyper.R(use_pandas=True)
for i in range(10):
    r("")


def main():
    fnamer1 = "log/all_6_560_result_20220710185619.csv"
    fnamer2 = "log/a2c_cnn_r4m11b_20220704124029.csv"
    df1 = pd.read_csv(fnamer1)
    df2 = pd.read_csv(fnamer2)[:1000]
    # print(df1)
    # print(df2)
    # return
    # print(df)
    r.assign("df1", df1)
    r.assign("df2", df2)
    # print(r('df["animals"]'))
    # welchのt検定
    moji = r("summary(df1)")
    moji += r("summary(df2)")
    moji += r('t.test(df1["animals"], df2["animals"])')
    print(moji)


if __name__ == "__main__":
    main()
