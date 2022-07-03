#!/usr/bin/env python3
"""
統計値計算
"""
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="学習の推移を描画")

parser.add_argument("file", help="読み込むファイル")

args = parser.parse_args()


def main(fnamer):
    move_mean_length = 50
    print(fnamer)
    df = pd.read_csv(fnamer)
    # print(df)
    x = df.index[:-move_mean_length+1]
    y1 = df["animals"]
    y2 = df["height"]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    y1_conv = np.convolve(y1, np.ones(move_mean_length) /
                          move_mean_length, mode='valid')
    y2_conv = np.convolve(y2, np.ones(move_mean_length) /
                          move_mean_length, mode='valid')
    # ax.plot(x, y1[:-move_mean_length+1])
    ax.plot(x, y1_conv, label="animals")
    ax.plot(x, y2_conv, label="height")
    ax.legend()
    ax.grid()
    # print(y1.shape)
    # print(y1_conv.shape)
    plt.show()


if __name__ == "__main__":
    main(args.file)
