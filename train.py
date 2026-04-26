"""
训练：数据加载、模型、训练循环、按验证集保存最优权重.
用法（在项目根目录）:
  python train.py
  python train.py --hidden 256 --lr 0.1 --l2 1e-4 --epochs 20 --save checkpoints/best
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.data_loader import FASHION_LABELS, load_datasets
from src.mlp import MLP
from src.train_utils import TrainConfig, eval_loss_acc, run_training

CLASS_NAMES = list(FASHION_LABELS)


def main() -> None:
    p = argparse.ArgumentParser(description="Fashion-MNIST NumPy MLP 训练")
    p.add_argument("--data_dir", type=str, default="data", help="原始 idx 数据目录")
    p.add_argument("--val_size", type=int, default=5000, help="从训练集划出的验证集大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层大小")
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--lr", type=float, default=0.1, help="初始学习率")
    p.add_argument("--l2", type=float, default=1e-3, help="L2/权重衰减系数 λ")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr_decay", type=float, default=0.95, help="每 epoch 后 lr 乘以该因子")
    p.add_argument("--no_lr_decay", action="store_true", help="关闭学习率衰减")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save",
        type=str,
        default="checkpoints/best",
        help="保存最优权重基路径(将生成 .npz 与 .meta.json)",
    )
    p.add_argument(
        "--history",
        type=str,
        default="results/train_history.json",
        help="训练曲线数据 JSON 路径(用于报告/画图)",
    )
    p.add_argument("--quiet", action="store_true", help="关闭 tqdm 进度条")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        hidden=args.hidden,
        activation=args.activation,
        lr_decay_gamma=args.lr_decay,
        step_decay=not args.no_lr_decay,
        seed=args.seed,
        data_dir=args.data_dir,
        val_size=args.val_size,
    )

    train, val, test = load_datasets(
        data_dir=cfg.data_dir, val_size=cfg.val_size, seed=cfg.seed
    )
    mlp = MLP(784, cfg.hidden, 10, activation=cfg.activation, seed=cfg.seed)

    history = run_training(
        mlp,
        train.x,
        train.y,
        val.x,
        val.y,
        cfg,
        quiet=args.quiet,
    )

    meta = {
        "d_in": 784,
        "d_h": cfg.hidden,
        "d_out": 10,
        "activation": cfg.activation,
        "lr": cfg.lr,
        "l2": cfg.l2,
        "lr_decay_gamma": cfg.lr_decay_gamma,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "val_size": cfg.val_size,
        "class_names": CLASS_NAMES,
    }
    MLP.save_to_npz(args.save, mlp, meta=meta)

    # 合并 history 为可 JSON 的 dict
    out_hist = {**history, "config": {**meta}}
    h_path = args.history
    if not os.path.isabs(h_path):
        h_path = os.path.join(ROOT, h_path)
    os.makedirs(os.path.dirname(h_path) or ".", exist_ok=True)
    with open(h_path, "w", encoding="utf-8") as f:
        json.dump(out_hist, f, ensure_ascii=False, indent=2)

    te_loss, te_acc = eval_loss_acc(
        mlp, test.x, test.y, cfg.l2, batch_size=512
    )
    print(
        f"\n[已保存权重] {args.save}.npz 与 {args.save}.meta.json | "
        f"验证集 best ≈ {history.get('best_val_acc', 0):.4f} | 测试(当前加载最优权重) loss={te_loss:.4f} acc={te_acc:.4f}"
    )
    print(f"[曲线数据] 已写 {h_path}，可运行: python plot_from_history.py --history {h_path}")


if __name__ == "__main__":
    main()
