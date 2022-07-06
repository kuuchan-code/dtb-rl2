#!/usr/bin/env python3
"""
統計値計算
"""
from __future__ import annotations
import pyper
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description="csvの結合")

parser.add_argument("file_dst", help="過去のデータ")
parser.add_argument("file_src", help="新規データ")

args = parser.parse_args()


def main(fname_dst, fname_src):
    print(fname_dst, fname_src)
    df_old = pd.read_csv(fname_dst)
    df_new = pd.read_csv(fname_src)
    print(df_old)
    print(df_new)


if __name__ == "__main__":
    main(args.file_dst, args.file_src)
