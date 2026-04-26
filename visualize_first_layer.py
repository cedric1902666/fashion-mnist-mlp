"""
将第一层权重 W1 的列(或行)恢复为 28x28 灰度图并保存栅格(报告中的「权重视觉化」).
常见做法: 每个隐藏神经元对应 784x1, 可 reshape 为 28x28; 本实现展示前 N 个隐藏元的输入权重列.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np

from src.mlp import MLP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best")
    ap.add_argument("--out", type=str, default="results/first_layer_weights.png")
    ap.add_argument("--n_show", type=int, default=25, help="展示前 n 个隐藏神经元的权重视图 (网格)")
    ap.add_argument(
        "--per_row", type=int, default=5, help="每行小图数量"
    )
    args = ap.parse_args()

    mlp, _ = MLP.load_from_npz(args.ckpt)
    # W1: (784, d_h) -> 第 j 列是第 j 个隐藏元对所有输入的权重, reshape(28,28)
    w = mlp.W1
    d_h = w.shape[1]
    n = min(args.n_show, d_h)
    ncols = min(args.per_row, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows)
    )
    axes = np.array(axes).reshape(-1)
    for j in range(n):
        im = w[:, j].reshape(28, 28)
        # 对比度: 按自身 min/max 归一化
        a, b = float(im.min()), float(im.max())
        if b - a < 1e-8:
            nrm = im * 0
        else:
            nrm = (im - a) / (b - a)
        ax = axes[j]
        ax.imshow(nrm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"h{j}", fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("First layer (input→hidden) weight columns, 28×28 each", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"已保存: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
