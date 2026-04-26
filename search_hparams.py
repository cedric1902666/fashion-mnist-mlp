"""
超参数搜索: 支持随机/网格. 用验证集选最优, 记录表格与 JSON, 可自动保存最优一次完整权重.
运行示例:
  python search_hparams.py --method random --trials 12 --epochs 5
  python search_hparams.py --method grid  --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.data_loader import FASHION_LABELS, load_datasets
from src.mlp import MLP
from src.train_utils import TrainConfig, run_training

CLASS_NAMES = list(FASHION_LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["random", "grid"], default="random")
    ap.add_argument("--trials", type=int, default=10, help="仅 random: 采样的组数")
    ap.add_argument("--epochs", type=int, default=8, help="每组训练的 epoch(搜参可设小一些)")
    ap.add_argument(
        "--save_best",
        type=str,
        default="checkpoints/hparam_best",
        help="用全局最优(按验证 acc)的权重与 meta 存于此基名",
    )
    ap.add_argument(
        "--log_csv",
        type=str,
        default="results/hparam_runs.csv",
    )
    ap.add_argument(
        "--log_json",
        type=str,
        default="results/hparam_summary.json",
    )
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_decay", type=float, default=0.95)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train, val, _ = load_datasets(
        data_dir=args.data_dir, val_size=args.val_size, seed=42
    )

    # 网格别太大(组合数=乘积). 可改大以换更全搜索(耗时更久)
    lrs = [0.1, 0.05, 0.01]
    hiddens = [128, 256, 512]
    l2s = [0.0, 1e-3]
    acts: List[str] = ["relu", "tanh"]

    configs: List[Dict[str, Any]] = []
    if args.method == "grid":
        for lr, h, l2, act in product(lrs, hiddens, l2s, acts):
            configs.append(
                {
                    "lr": lr,
                    "hidden": h,
                    "l2": l2,
                    "activation": act,
                }
            )
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.trials):
            configs.append(
                {
                    "lr": float(rng.choice(lrs)),
                    "hidden": int(rng.choice(hiddens)),
                    "l2": float(rng.choice(l2s)),
                    "activation": str(rng.choice(acts)),
                }
            )

    best_val = -1.0
    best_row: Dict[str, Any] | None = None
    best_mlp: MLP | None = None
    all_rows: List[Dict[str, Any]] = []

    for i, c in enumerate(configs):
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=c["lr"],
            l2=c["l2"],
            hidden=c["hidden"],
            activation=c["activation"],
            lr_decay_gamma=args.lr_decay,
            data_dir=args.data_dir,
            val_size=args.val_size,
        )
        m = MLP(784, cfg.hidden, 10, activation=cfg.activation, seed=args.seed + i)
        h = run_training(
            m, train.x, train.y, val.x, val.y, cfg, quiet=True
        )
        va = h["best_val_acc"]
        row = {
            "idx": i,
            "lr": c["lr"],
            "hidden": c["hidden"],
            "l2": c["l2"],
            "activation": c["activation"],
            "best_val_acc": va,
        }
        all_rows.append(row)
        if va > best_val:
            best_val = va
            best_row = {**row, "history": h}
            best_mlp = m
        print(
            f"[{i+1}/{len(configs)}] "
            f"lr={c['lr']} h={c['hidden']} l2={c['l2']} act={c['activation']}  ->  val={va:.4f}"
        )

    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["idx", "lr", "hidden", "l2", "activation", "best_val_acc"],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(
                {k: r.get(k) for k in w.fieldnames if k in r}  # type: ignore
            )

    with open(args.log_json, "w", encoding="utf-8") as f:
        best_for_dump = None
        if best_row is not None:
            best_for_dump = {k: v for k, v in best_row.items() if k != "history"}
        json.dump(
            {
                "best": best_for_dump,
                "all": all_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if best_mlp is not None and best_row is not None:
        br = {k: v for k, v in best_row.items() if k != "history"}
        meta = {
            "d_in": 784,
            "d_h": int(best_row["hidden"]),
            "d_out": 10,
            "activation": str(best_row["activation"]),
            "lr": float(best_row["lr"]),
            "l2": float(best_row["l2"]),
            "search_method": args.method,
            "epochs_in_search": args.epochs,
            "best_val_acc": best_row["best_val_acc"],
            "class_names": CLASS_NAMES,
        }
        MLP.save_to_npz(args.save_best, best_mlp, meta=meta)
    print(
        f"\n已写入 {args.log_csv} 与 {args.log_json} | "
        f"当前搜索内最优 val_acc = {best_val:.4f} | 权重(搜参) -> {args.save_best}.npz"
    )
    if best_row is not None:
        print(
            "若需以最优超参再「完整训练长 epoch」, 可手工运行:\n"
            f"  python train.py --lr {best_row['lr']} "
            f"--hidden {best_row['hidden']} --l2 {best_row['l2']} "
            f"--activation {best_row['activation']} --epochs 25 --save checkpoints/best"
        )


if __name__ == "__main__":
    main()
