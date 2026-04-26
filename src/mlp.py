"""
三层 MLP（一层隐藏层）+ 手算反向传播。仅使用 NumPy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

ActivationName = Literal["relu", "sigmoid", "tanh"]


def _softmax_logsumexp(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


@dataclass
class BackwardCache:
    x: np.ndarray
    z1: np.ndarray
    h1: np.ndarray
    z2: np.ndarray
    act_name: str


class MLP:
    """
    结构: 输入 d_in -> 隐藏 d_h (激活) -> 输出 d_out (logits)
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        d_out: int,
        activation: ActivationName = "relu",
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        self.activation = activation
        rng = np.random.default_rng(seed)
        # He / Xavier: 用 fan_in
        s1 = np.sqrt(2.0 / d_in)
        s2 = np.sqrt(2.0 / d_h)
        self.W1 = (rng.standard_normal((d_in, d_h)) * s1).astype(np.float32)
        self.b1 = np.zeros((1, d_h), dtype=np.float32)
        self.W2 = (rng.standard_normal((d_h, d_out)) * s2).astype(np.float32)
        self.b2 = np.zeros((1, d_out), dtype=np.float32)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.W1 = np.array(params["W1"], dtype=np.float32, copy=True)
        self.b1 = np.array(params["b1"], dtype=np.float32, copy=True)
        self.W2 = np.array(params["W2"], dtype=np.float32, copy=True)
        self.b2 = np.array(params["b2"], dtype=np.float32, copy=True)

    def _act_f(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        if self.activation == "tanh":
            return np.tanh(z)
        raise ValueError(self.activation)

    def _act_d(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return (h * (1.0 - h)).astype(np.float32)
        if self.activation == "tanh":
            return (1.0 - h * h).astype(np.float32)
        raise ValueError(self.activation)

    def forward(self, x: np.ndarray, store_cache: bool = True) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        h1 = self._act_f(z1)
        z2 = h1 @ self.W2 + self.b2
        if store_cache:
            self._cache = BackwardCache(
                x=x, z1=z1, h1=h1, z2=z2, act_name=self.activation
            )
        return z2

    def mean_cross_entropy_l2(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        l2: float,
    ) -> float:
        b = logits.shape[0]
        p = _softmax_logsumexp(logits)
        y = y.astype(np.int64, copy=False)
        logp = -np.log(p[np.arange(b), y] + 1e-12)
        ce = float(np.mean(logp))
        reg = 0.5 * l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return ce + reg

    def backward_batch(
        self,
        y: np.ndarray,
        l2: float,
    ) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_cache"):
            raise RuntimeError("需先 forward(x)")
        c = self._cache
        x, z1, h1, z2 = c.x, c.z1, c.h1, c.z2
        b = x.shape[0]
        p = _softmax_logsumexp(z2)
        y = y.astype(np.int64, copy=False)
        one = np.zeros_like(p, dtype=np.float32)
        one[np.arange(b), y] = 1.0
        d_z2 = (p - one) / float(b)
        dW2 = h1.T @ d_z2 + l2 * self.W2
        db2 = np.sum(d_z2, axis=0, keepdims=True)
        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * self._act_d(z1, h1)
        dW1 = x.T @ d_z1 + l2 * self.W1
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    @staticmethod
    def save_to_npz(
        path: str,
        mlp: "MLP",
        meta: Optional[dict] = None,
    ) -> None:
        import json
        import os

        base = path
        if base.endswith(".npz"):
            base = base[:-4]
        out_npz = base if path.endswith(".npz") else path
        if not out_npz.endswith(".npz"):
            out_npz = out_npz + ".npz"
        p = mlp.get_params()
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez(
            out_npz,
            W1=p["W1"],
            b1=p["b1"],
            W2=p["W2"],
            b2=p["b2"],
            d_in=np.int32(mlp.d_in),
            d_h=np.int32(mlp.d_h),
            d_out=np.int32(mlp.d_out),
        )
        full_meta = {
            "activation": mlp.activation,
        }
        if meta:
            full_meta.update(meta)
        with open(base + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(full_meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_npz(cls, path: str) -> Tuple["MLP", dict]:
        import json
        import os

        p = path if path.endswith(".npz") else path + ".npz"
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"未找到权重文件: {p}。请使用 train.py 训练或从网盘下载后放到 checkpoints/"
            )
        z = np.load(p, allow_pickle=False)
        d_in, d_h, d_out = int(z["d_in"]), int(z["d_h"]), int(z["d_out"])
        act: ActivationName = "relu"  # type: ignore
        meta: dict = {}
        base = p[:-4] if p.endswith(".npz") else p
        meta_p = base + ".meta.json"
        if os.path.isfile(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            a = meta.get("activation", "relu")
            if a in ("relu", "sigmoid", "tanh"):
                act = a  # type: ignore
        m = cls(d_in, d_h, d_out, activation=act)  # type: ignore
        m.set_params({k: np.array(z[k], dtype=np.float32) for k in ("W1", "b1", "W2", "b2")})
        return m, meta
