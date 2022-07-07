#!/usr/bin/env python3
"""
統計値計算
"""
from __future__ import annotations
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="csvの結合")

parser.add_argument("file_dst", help="過去のデータ")
parser.add_argument("file_src", help="新規データ")

args = parser.parse_args()


def main(fname_dst, fname_src):
    res = input(f"{fname_src}は削除され, {fname_dst}は上書きされます. よろしいですか? (y/n):")
    if res != "y":
        return -1
    print(fname_dst, fname_src)
    df_old = pd.read_csv(fname_dst)
    df_new = pd.read_csv(fname_src)
    df_all = pd.concat([df_old, df_new])
    print(df_all)
    # df_all.to_csv("test.csv", index=False)
    # ファイル結合
    df_all.to_csv(fname_dst, index=False)
    # 削除
    os.remove(fname_src)
    return 0


if __name__ == "__main__":
    main(args.file_dst, args.file_src)
