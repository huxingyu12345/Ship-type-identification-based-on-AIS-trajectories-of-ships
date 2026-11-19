# 船舶 AIS 轨迹船型识别

基于船舶 AIS 轨迹（经纬度、航速、航向、时间戳），识别四类船型：

- 集装箱船
- 干散货船
- 渔船
- 油船

评价指标为 **准确率**。提交文件为 UTF-8 编码的 `result.csv`。

## 赛题要点

| 项目 | 说明 |
| --- | --- |
| 训练集 | 3600 艘，4 类各 900 艘，时间跨度约 8 个月 |
| 测试集 | 300 艘，无标签，时间跨度约 1 个月 |
| 输入字段 | `shipno, lat, lon, sog, cog, postime` |
| 提交格式 | `shipno,shiptype`，编码 UTF-8 |

训练集与测试集时间跨度不同。特征工程应尽量避免只依赖轨迹时长、点数等时序敏感量。本项目通过把 8 个月训练轨迹切成 1/2/4/8 个月窗口，让模型也能见到接近测试集长度的样本。

## 方法概述

1. **时间切分**：将训练集按 1/2/4/8 个月多粒度切片，缓解训练/测试跨度不一致。
2. **表格特征**：从轨迹提取航速统计、曲率、停留、直线度、网格熵、FFT 等运动特征。
3. **轨迹图像**：把经纬度轨迹画成 224×224 图（颜色编码时间，线宽编码航速）。
4. **图像特征**：用 ImageNet 预训练的 EfficientNetB3 抽取图像向量。
5. **特征选择**：表格特征用 XGBoost RFE 保留 60 维，图像特征按重要性保留 20 维，再做多项式交互。
6. **混合网络**：表格分支 + 图像分支 + Multi-Head Attention，5 折交叉验证，取验证最好的一折预测测试集。

## 目录结构

```text
xx/
├── main.py                 # 统一入口
├── data_split.py           # 兼容旧命令：只跑时间切分
├── train/                  # 有标签训练轨迹
│   ├── 干散货船/           # 900 艘
│   ├── 渔船/               # 900 艘
│   ├── 油船/               # 900 艘
│   └── 集装箱船/           # 900 艘
├── test/                   # 无标签测试轨迹，300 艘
├── src/
│   ├── config.py           # 路径与超参数
│   ├── pipeline.py         # 切分 / 特征 / 训练 / 预测串联
│   ├── data/               # 读数据、去噪、时间切分
│   ├── features/           # 表格特征、轨迹绘图、CNN 特征
│   ├── models/             # 混合网络、XGBoost 特征选择
│   └── utils/              # 时间戳解析、距离计算、日志
└── outputs/                # 运行后自动生成
    ├── train_split/        # 多粒度切分数据
    ├── train_images/       # 训练轨迹图
    ├── test_images/        # 测试轨迹图
    ├── features/           # 表格特征缓存
    ├── logs/               # 运行日志
    └── result.csv          # 提交文件
```

每个 CSV 对应一艘船。`postime` 为 Unix 秒级时间戳。

## 环境依赖

建议 Python 3.8+，并安装：

```bash
pip install numpy pandas scikit-learn scipy tqdm pillow xgboost tensorflow
```

说明：

- 表格特征提取需要 `scipy`、`scikit-learn`。
- 图像分支需要 `tensorflow`（EfficientNetB3 + 混合网络）。
- 默认 XGBoost 使用 `gpu_hist`，GPU 编号为 `1`。若本机没有对应 GPU，请修改 `src/config.py` 中的 `xgb_tree_method` 和 `xgb_gpu_id`，例如改成 `hist`。

## 快速开始

在项目根目录运行：

```bash
python main.py
```

默认 `--step all`：若还没有切分数据，会先切分训练集，再提取特征、训练并写出 `outputs/result.csv`。

分步执行：

```bash
# 只按时间切分训练集
python main.py --step split

# 已有切分结果时，只训练和预测
python main.py --step train

# 强制重新切分后再训练
python main.py --force-split

# 修改时间切分份数（默认 8）
python main.py --num-splits 8
```

兼容旧入口：

```bash
python data_split.py
python train.py
```

特征 CSV 和轨迹图会缓存到 `outputs/`。再次训练时若缓存存在会直接读取；若要重做特征，删掉对应文件即可。

## 提交文件

路径：`outputs/result.csv`

```text
shipno,shiptype
0a0d8ebc,干散货船
```

船型标签必须是：`集装箱船`、`干散货船`、`渔船`、`油船`。请保持 UTF-8 编码，避免评测因中文编码失败。

## 主要配置

路径和超参数集中在 `src/config.py`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `train_dir` / `test_dir` | `train/`、`test/` | 原始数据目录 |
| `output_dir` | `outputs/` | 全部中间结果和提交文件 |
| `num_splits` | 8 | 时间切分份数 |
| `rfe_n_features` | 60 | 表格特征 RFE 保留维数 |
| `image_top_k` | 20 | 图像特征保留维数 |
| `n_folds` | 5 | 交叉验证折数 |
| `epochs` | 100 | 训练轮数，配合 early stopping |
| `xgb_tree_method` | `gpu_hist` | XGBoost 计算后端 |
| `xgb_gpu_id` | 1 | XGBoost 使用的 GPU 编号 |

## 流水线

```text
train/ + test/
        │
        ▼
  时间切分 (1/2/4/8 个月)
        │
        ├──────────────┐
        ▼              ▼
  表格特征提取     轨迹图像生成
        │              │
        │              ▼
        │        EfficientNetB3
        │              │
        ▼              ▼
     RFE 选 60 维    重要性选 20 维
        │              │
        └──────┬───────┘
               ▼
        多项式交互 + 拼接
               ▼
     混合网络 5 折训练
               ▼
        outputs/result.csv
```

## 常见问题

**切分很慢、文件很多**  
切分会按船、按时间窗口写出大量 CSV，这是预期行为。完成后会缓存在 `outputs/train_split/`。

**特征提取占用 CPU 很高**  
表格特征和轨迹图使用多进程，进程数由 CPU 核数决定，上限 10。可在 `src/config.py` 的 `n_jobs` 里调小。

**XGBoost / TensorFlow 报 GPU 错误**  
把 `xgb_tree_method` 改为 `hist`，并确认本机 TensorFlow 能使用的设备。混合网络本身用 TensorFlow 训练。

**想重新跑某一阶段**  
- 重切分：`python main.py --force-split`
- 重提表格特征：删除 `outputs/features/`
- 重画轨迹图：删除 `outputs/train_images/` 或 `outputs/test_images/`
