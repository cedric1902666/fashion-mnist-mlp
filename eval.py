"""
在官方测试集上评估已保存的权重: 总体准确率 + 混淆矩阵.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.data_loader import FASHION_LABELS, load_datasets
from src.mlp import MLP
from src.train_utils import eval_loss_acc


def confusion_matrix(prod: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    c = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y, prod):
        c[t, p] += 1
    return c


def print_confusion(c: np.ndarray, names) -> None:
    print("\n混淆矩阵 (行: 真值, 列: 预测)")
    w = max(6, max(len(n) for n in names) + 1)
    head = " " * w + "".join(f"{i:>{w - 1}} " for i in range(c.shape[1]))
    print(head)
    for i, row in enumerate(c):
        print(f"{i} {names[i]:<{w-2}} " + " ".join(f"{v:>5}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=str, default="checkpoints/best", help="不含 .npz 或含 .npz 均可"
    )
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--val_size", type=int, default=5000, help="需与训练时一致(仅影响 train 划分, test 为官方 10k)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l2", type=float, default=None, help="若未指定, 从 .meta.json 的 l2 读取")
    args = ap.parse_args()

    mlp, meta = MLP.load_from_npz(args.ckpt)
    l2 = args.l2 if args.l2 is not None else float(meta.get("l2", 0.0))

    _, _, test = load_datasets(
        data_dir=args.data_dir, val_size=args.val_size, seed=args.seed
    )
    loss, acc = eval_loss_acc(mlp, test.x, test.y, l2, batch_size=512)
    logits_all = None
    for s in range(0, test.x.shape[0], 512):
        t = s + 512
        z = mlp.forward(test.x[s:t], store_cache=False)
        logits_all = z if logits_all is None else np.vstack((logits_all, z))
    assert logits_all is not None
    pred = np.argmax(logits_all, axis=1)
    cm = confusion_matrix(pred, test.y, 10)
    print(f"测试集: loss(含L2)={loss:.4f}  |  accuracy={acc:.4f}\n")
    print_confusion(cm, FASHION_LABELS)


if __name__ == "__main__":
    main()
