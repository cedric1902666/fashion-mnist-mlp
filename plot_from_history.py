"""
从 train.py 保存的 JSON 画训练/验证 Loss 与验证 Accuracy 曲线（报告用）。
使用纯英文标签与显式 x 轴刻度，避免 Matplotlib 数学解析与单点缩放问题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 禁止把括号内容当数学公式解析
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.default"] = "regular"


def _xtick_epochs(n: int) -> list[int]:
    if n <= 1:
        return [1]
    if n <= 20:
        return list(range(1, n + 1))
    # 长训练: 少标一些刻度
    step = max(1, n // 12)
    return list(range(1, n + 1, step)) + [n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history",
        type=str,
        default="results/train_history.json",
    )
    ap.add_argument("--out", type=str, default="results/curves.png")
    args = ap.parse_args()

    with open(args.history, "r", encoding="utf-8") as f:
        data = json.load(f)
    t_loss = data["train_loss"]
    v_loss = data["val_loss"]
    v_acc = data["val_acc"]
    n = len(t_loss)
    if n != len(v_loss) or n != len(v_acc):
        raise SystemExit("train_loss/val_loss/val_acc 长度不一致，请检查 JSON。")
    if n < 1:
        raise SystemExit("没有 epoch 数据。")

    ep = np.arange(1, n + 1, dtype=int)
    print(f"Read {n} epochs from JSON; plotting x=1..{n}.")

    # constrained_layout 减少 twinx 被裁切
    fig, ax1 = plt.subplots(figsize=(9, 4.8), layout="constrained")
    # 全 ASCII，避免 $...$ 与缺字
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlim(0.5, n + 0.5)
    ax1.set_xticks(_xtick_epochs(n))
    ax1.minorticks_on()
    ax1.grid(True, which="major", alpha=0.35)
    ax1.plot(ep, t_loss, label="train loss", color="C0", marker="o", ms=3, lw=1.2)
    ax1.plot(ep, v_loss, label="val loss", color="C1", marker="s", ms=3, lw=1.2)
    # 在图下注明损失定义（不放进 ylabel 避免解析问题）
    fig.suptitle(
        "Fashion-MNIST MLP: mean CE plus L2 weight decay, per epoch",
        fontsize=10,
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("val accuracy", color="C2")
    ax2.set_xlim(0.5, n + 0.5)
    ax2.plot(
        ep,
        v_acc,
        color="C2",
        lw=2,
        marker="^",
        ms=3,
        label="val acc",
    )
    ax2.tick_params(axis="y", labelcolor="C2")
    y2 = np.asarray(v_acc, dtype=float)
    pad = 0.02
    lo2, hi2 = float(y2.min()), float(y2.max())
    if hi2 - lo2 < 0.01:
        ax2.set_ylim(lo2 - 0.02, hi2 + 0.02)
    else:
        ax2.set_ylim(lo2 - (hi2 - lo2) * pad, hi2 + (hi2 - lo2) * pad)

    lines, labels_ = ax1.get_legend_handles_labels()
    lines2, labels2_ = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_ + labels2_, loc="center right", fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
