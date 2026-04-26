from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .mlp import MLP


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.1
    l2: float = 1e-4
    hidden: int = 256
    activation: str = "relu"
    lr_decay_gamma: float = 0.95
    step_decay: bool = True  # 每 epoch 后 lr *= gamma
    seed: int = 42
    data_dir: str = "data"
    val_size: int = 5000


def iter_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = idx.copy()
        rng.shuffle(idx)
    batches = []
    for s in range(0, n, batch_size):
        t = min(s + batch_size, n)
        batches.append((x[idx[s:t]], y[idx[s:t]]))
    return batches


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def sgd_step(mlp: MLP, grads: Dict[str, np.ndarray], lr: float) -> None:
    mlp.W1 -= lr * grads["W1"]
    mlp.b1 -= lr * grads["b1"]
    mlp.W2 -= lr * grads["W2"]
    mlp.b2 -= lr * grads["b2"]


def train_one_epoch(
    mlp: MLP,
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2: float,
    batch_size: int,
    seed: int,
) -> float:
    batches = iter_minibatches(x, y, batch_size, shuffle=True, seed=seed)
    losses: List[float] = []
    for xb, yb in batches:
        logits = mlp.forward(xb, store_cache=True)
        loss = mlp.mean_cross_entropy_l2(logits, yb, l2)
        g = mlp.backward_batch(yb, l2)
        sgd_step(mlp, g, lr)
        losses.append(loss)
    return float(np.mean(losses)) if losses else 0.0


def _ce_only(mlp: MLP, logits: np.ndarray, y: np.ndarray) -> float:
    from .mlp import _softmax_logsumexp

    p = _softmax_logsumexp(logits)
    b = logits.shape[0]
    y = y.astype(np.int64, copy=False)
    logp = -np.log(p[np.arange(b), y] + 1e-12)
    return float(np.mean(logp))


def eval_loss_acc(
    mlp: MLP, x: np.ndarray, y: np.ndarray, l2: float, batch_size: int = 256
) -> Tuple[float, float]:
    n = x.shape[0]
    ce_acc = 0.0
    n_seen = 0
    for s in range(0, n, batch_size):
        t = min(s + batch_size, n)
        xb, yb = x[s:t], y[s:t]
        logits = mlp.forward(xb, store_cache=False)
        ce_acc += _ce_only(mlp, logits, yb) * (t - s)
        n_seen += t - s
    ce_mean = ce_acc / max(n_seen, 1)
    reg = 0.5 * l2 * (np.sum(mlp.W1 * mlp.W1) + np.sum(mlp.W2 * mlp.W2))
    loss = float(ce_mean + reg)
    all_logits: Optional[np.ndarray] = None
    for s in range(0, n, batch_size):
        t = min(s + batch_size, n)
        lo = mlp.forward(x[s:t], store_cache=False)
        all_logits = lo if all_logits is None else np.vstack((all_logits, lo))
    assert all_logits is not None
    acc = accuracy(all_logits, y)
    return loss, acc


def run_training(
    mlp: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    quiet: bool = False,
) -> dict:
    """
    按 epoch 训练，学习率每 epoch 衰减 (若 cfg.step_decay).
    返回 history 字典: train_loss, val_loss, val_acc, lrs, best_val_acc, best_state 路径由调用方写盘.
    """
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    lr = float(cfg.lr)
    l2 = float(cfg.l2)
    best_val = -1.0
    best_state: Optional[Dict[str, np.ndarray]] = None
    pbar = range(cfg.epochs)
    if not quiet:
        pbar = tqdm(pbar, desc="epochs", ncols=88)
    for ep in pbar:
        tr_loss = train_one_epoch(
            mlp,
            x_train,
            y_train,
            lr=lr,
            l2=l2,
            batch_size=cfg.batch_size,
            seed=cfg.seed + ep,
        )
        va_loss, va_acc = eval_loss_acc(
            mlp, x_val, y_val, l2, batch_size=512
        )
        if cfg.step_decay and ep < cfg.epochs - 1:
            lr *= cfg.lr_decay_gamma
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr)
        if va_acc > best_val + 1e-12:
            best_val = va_acc
            best_state = {k: v.copy() for k, v in mlp.get_params().items()}
        if not quiet and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                tr=f"{tr_loss:.4f}",
                va=f"{va_acc:.4f}",
                lr=f"{lr:.5f}",
            )
    if best_state is not None:
        mlp.set_params(best_state)
    history["best_val_acc"] = float(best_val)
    return history