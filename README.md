# fashion-mnist-mlp

NumPy 手写三层 MLP + 反向传播，[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 十类分类。课程 HW1。

## 环境

```text
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.10+ 推荐。首次训练会自动下数据到 `data/`；断网时按官方仓库说明把四个 `.gz` 放进 `data/`。

## 权重

仓库不含 `checkpoints/*.npz`。作业里给的网盘链接下载 `best.npz`、`best.meta.json` 放到 `checkpoints/`。

## 常用命令

```text
python train.py --epochs 20 --save checkpoints/best
python eval.py --ckpt checkpoints/best
python search_hparams.py --method random --trials 10 --epochs 8
python plot_from_history.py
python visualize_first_layer.py --ckpt checkpoints/best
python error_analysis.py --ckpt checkpoints/best
```

`train.py` 可选参数示例：`--hidden`、`--lr`、`--l2`、`--activation relu|sigmoid|tanh`、`--lr_decay`。

## 代码大致对应

- `src/data_loader.py`：数据  
- `src/mlp.py`：模型与反传  
- `src/train_utils.py`：训练与 SGD  
- 根目录脚本：训练 / 测试 / 搜参 / 画图
