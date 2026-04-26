# fashion-mnist-mlp

Computer Vision HW1: 仅使用 **NumPy** 实现三层 MLP（一层隐藏层）、手写反向传播，在 [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 上完成 10 类分类。

## 预训练权重（作业提交）

本仓库不附带 `checkpoints/*.npz`；请从**实验报告 / 课程要求**中给出的 **Google Drive（或其它网盘）链接** 下载 `best.npz` 与 `best.meta.json`，放入项目下 `checkpoints/`，再运行 `eval.py` 等。链接由提交者维护。

## 环境

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

建议 Python 3.10+。

## 目录说明

| 路径 | 说明 |
|------|------|
| `src/data_loader.py` | 下载/读取数据、划分训练/验证、预处理 |
| `src/mlp.py` | MLP、Softmax+交叉熵、L2、反向传播、权重 `npz` 存读 |
| `src/train_utils.py` | SGD、学习率衰减、训练/验证、指标 |
| `train.py` | 单次正式训练，保存**验证集最优**权重与 `results/train_history.json` |
| `search_hparams.py` | 网格或随机超参搜索，记录 `results/hparam_*.` |
| `eval.py` | 加载权重，在**官方 10k 测试集**上输出准确率 + **混淆矩阵** |
| `plot_from_history.py` | 从 JSON 画 **Loss/验证 Accuracy** 曲线图（放报告用） |
| `visualize_first_layer.py` | **第一层权重视觉化**（28×28 小图，放报告用） |
| `error_analysis.py` | 导出部分**错分样本**拼图（放报告用） |
| `data/` | 自动下载的原始数据（不提交时可在 `.gitignore` 中忽略） |
| `checkpoints/` | 训练得到的 `.npz` + `.meta.json`（**大文件/权重**建议不提交，改传网盘） |
| `results/` | 曲线图、权重视觉化、错例图、训练历史 JSON |

## 数据下载

第一次运行 `train.py` 时会从官方 S3 自动下载四个 `.gz` 到 `data/`。若网络失败，请从 [Fashion-MNIST 仓库](https://github.com/zalandoresearch/fashion-mnist) 说明中手动下载同名字文件，仍放入 `data/`。

## 推荐流程

### 1) 先完整训练一版（得到权重与报告用曲线）

```bash
python train.py --epochs 20 --save checkpoints/best
```

可调整：`--hidden`、`--lr`、`--l2`、`--activation {relu,sigmoid,tanh}`、`--lr_decay`、`--no_lr_decay`。

训练结束后会生成：

- `checkpoints/best.npz`、 `checkpoints/best.meta.json`：最优权重 + 元数据（把这两个**打包上传网盘**即可，见下）
- `results/train_history.json`：每个 epoch 的 `train_loss` / `val_loss` / `val_acc`

### 2) 画报告要求的曲线

```bash
python plot_from_history.py --history results/train_history.json --out results/curves.png
```

把 `results/curves.png` 插入 **PDF 实验报告**。

### 3) 测试集表现与混淆矩阵

```bash
python eval.py --ckpt checkpoints/best
```

### 4) 第一层权重视觉化

```bash
python visualize_first_layer.py --ckpt checkpoints/best --out results/first_layer_weights.png
```

### 5) 错例分析用图

```bash
python error_analysis.py --ckpt checkpoints/best --out results/error_samples.png
```

### 6) 超参搜索（作业要求：网格或随机）

```bash
# 随机搜索，每组 epoch 可设小一些以省时间
python search_hparams.py --method random --trials 12 --epochs 8 --save_best checkpoints/hparam_best
```

- 结果表：`results/hparam_runs.csv`、`results/hparam_summary.json`  
- 根据表格里**验证集**最高的行，可再用 `train.py` 用相同超参跑更长 epoch 得到最终 `checkpoints/best.*`。

## 把权重/结果「放到该放的地方」

| 要交的内容 | 建议做法 |
|------------|----------|
| **代码** | 把整个项目推到你的 **公开 GitHub**；确保 `README` 与作业要求说明的环境、命令一致。 |
| **训练好的模型权重** | 在资源管理器中选中 `checkpoints/best.npz` 和 `checkpoints/best.meta.json`（可打成 **一个 zip**），上传到 **Google Drive / 网盘**；网盘点「获取链接」设任何知道链接可查看。 |
| **实验报告 PDF** | 正文中应包含：**GitHub 链接**、**网盘里权重/压缩包的链接**、以及 `curves.png`、`first_layer_weights.png`、`error_samples.png` 等图（在 Word/LaTeX 里引用 `results/` 下文件）。 |
| **别上传太大文件到 Git** | `.gitignore` 已忽略 `data/`、`checkpoints/`、部分 `results/`，一般不要把几百 MB 数据或网盘能放的权重强推上 Git。 |

> 你已有仓库：把本机 `fashion-mnist-mlp` 目录里的代码同步到 [cedric1902666/fashion-mnist-mlp](https://github.com/cedric1902666/fashion-mnist-mlp) 即可，例如：在本目录 `git init` 若需要、`git remote add`、然后 `add/commit/push`（或复制进已 clone 的仓库再推送）。

## 作业对应关系（速查）

- 自实现反传、SGD、交叉熵、L2、学习率衰减、按验证集保存最优：`src/mlp.py`、`src/train_utils.py`、`train.py`  
- 五模块：数据 / 模型 / 训练 / 测试评估 / 超参：对应 `data_loader` + 上述脚本与 `search_hparams.py`  
- 报告图：用 `plot_from_history.py`、`visualize_first_layer.py`、`error_analysis.py` 出图

## 许可

仅用于课程作业提交；数据版权归 Fashion-MNIST 原项目说明。
