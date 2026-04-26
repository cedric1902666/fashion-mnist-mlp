"""
从测试集中选若干错分样本, 保存为一张拼图，便于在报告中做错因分析.
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

from src.data_loader import FASHION_LABELS, load_datasets
from src.mlp import MLP

CLASS_NAMES = list(FASHION_LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best")
    ap.add_argument("--n", type=int, default=8, help="展示错例个数(最多取这么多)")
    ap.add_argument(
        "--out", type=str, default="results/error_samples.png", help="输出图片路径"
    )
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mlp, _ = MLP.load_from_npz(args.ckpt)
    _, _, test = load_datasets(
        data_dir=args.data_dir, val_size=args.val_size, seed=args.seed
    )
    rng = np.random.default_rng(123)
    logits = mlp.forward(test.x, store_cache=False)
    pred = np.argmax(logits, axis=1)
    wrong_idx = np.where(pred != test.y)[0]
    if len(wrong_idx) == 0:
        print("无错分样本(当前模型在测试集上全对?)")
        return
    pick = wrong_idx
    if len(pick) > args.n:
        pick = rng.choice(pick, size=args.n, replace=False)

    n = len(pick)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    # 无整图大标题 (suptitle)；仅每格小标题 (true / pred)
    w_inch, h_inch = 1.55 * ncols, 1.85 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(w_inch, h_inch),
        layout="constrained",
    )
    axes = np.array(axes).reshape(-1)
    for k, ix in enumerate(pick):
        im = test.x[ix].reshape(28, 28)
        t, p = int(test.y[ix]), int(pred[ix])
        ax = axes[k]
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, aspect="equal")
        short = 10
        ax.set_title(
            f"true: {CLASS_NAMES[t][:short]}\npred: {CLASS_NAMES[p][:short]}",
            fontsize=6,
            pad=5,
        )
        ax.set_axis_off()
    for k in range(n, len(axes)):
        axes[k].set_axis_off()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", pad_inches=0.12)
    print("与图中顺序对应的样本 (测试集下标):")
    for k, ix in enumerate(pick):
        t, p = int(test.y[ix]), int(pred[ix])
        print(f"  #{k+1}  test_idx={ix}  true={CLASS_NAMES[t]}  pred={CLASS_NAMES[p]}")
    print(
        f"已保存 {n} 个错例到 {os.path.abspath(args.out)}；测试集总错分率: "
        f"{(pred != test.y).mean():.2%} ({(pred != test.y).sum()}/{len(test.y)})"
    )


if __name__ == "__main__":
    main()
