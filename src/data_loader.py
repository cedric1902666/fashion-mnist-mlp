"""
Fashion-MNIST 数据加载与预处理
数据集: https://github.com/zalandoresearch/fashion-mnist
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# 官方镜像 (若失败可将文件手动放入 data_dir，文件名见下)
BASE_URLS: List[str] = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
]


@dataclass
class DatasetSplit:
    x: np.ndarray  # (N, 784) float32, 在 [0,1] 或标准化后
    y: np.ndarray  # (N,) int64, 0..9


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
        magic, n = struct.unpack(">ii", buf[:8])
        if magic != 2049:
            raise ValueError(f"label magic mismatch: {magic}")
        y = np.frombuffer(buf[8:], dtype=np.uint8).copy()
    return y.astype(np.int64)


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
        magic, n, r, c = struct.unpack(">iiii", buf[:16])
        if magic != 2051:
            raise ValueError(f"image magic mismatch: {magic}")
        im = np.frombuffer(buf[16:], dtype=np.uint8).copy().reshape(n, r, c)
    return im


def _download(url: str, dest: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except OSError:
        return False


def _ensure_fashion_mnist_file(name: str, data_dir: str) -> str:
    out = os.path.join(data_dir, name)
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    os.makedirs(data_dir, exist_ok=True)
    for base in BASE_URLS:
        url = f"{base.rstrip('/')}/{name}"
        if _download(url, out) and os.path.getsize(out) > 0:
            return out
    raise FileNotFoundError(
        f"无法下载 {name}。请从 https://github.com/zalandoresearch/fashion-mnist "
        f"或镜像手动下载，放入: {os.path.abspath(data_dir)}"
    )


def load_fashion_mnist_raw(
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = {
        "train_x": "train-images-idx3-ubyte.gz",
        "train_y": "train-labels-idx1-ubyte.gz",
        "test_x": "t10k-images-idx3-ubyte.gz",
        "test_y": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {k: _ensure_fashion_mnist_file(v, data_dir) for k, v in files.items()}

    train_im = _read_images(paths["train_x"])
    train_y = _read_labels(paths["train_y"])
    test_im = _read_images(paths["test_x"])
    test_y = _read_labels(paths["test_y"])
    return train_im, train_y, test_im, test_y


def preprocess_images(
    x: np.ndarray, normalize: str = "scale"
) -> np.ndarray:
    """
    x: (N, 28, 28) 或 (N, 784) uint8
    normalize: "scale" -> /255, "none" 仅转 float
    """
    if x.ndim == 3:
        n = x.shape[0]
        x = x.reshape(n, -1)
    x = x.astype(np.float32)
    if normalize == "scale":
        x /= 255.0
    return x


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_size: int,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    assert len(x) == len(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    if val_size >= n:
        raise ValueError("val_size 过大")
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx])


def load_datasets(
    data_dir: str = "data",
    val_size: int = 5000,
    seed: int = 42,
    normalize: str = "scale",
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    train_im, train_y, test_im, test_y = load_fashion_mnist_raw(data_dir)
    x_all = preprocess_images(train_im, normalize=normalize)
    (x_tr, y_tr), (x_val, y_val) = train_val_split(
        x_all, train_y, val_size=val_size, seed=seed
    )
    x_test = preprocess_images(test_im, normalize=normalize)
    y_test = test_y
    return (
        DatasetSplit(x_tr, y_tr),
        DatasetSplit(x_val, y_val),
        DatasetSplit(x_test, y_test),
    )


# 10 个类别名（与官方顺序一致，便于错例/混淆矩阵标题）
FASHION_LABELS: Tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
