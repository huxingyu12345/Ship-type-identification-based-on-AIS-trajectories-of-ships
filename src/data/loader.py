import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import SHIP_TYPES, SPLIT_TYPES, TRAJECTORY_COLS, Config
from src.utils.common import parse_postime


def load_train_data(train_dir) -> pd.DataFrame:
    train_dir = Path(train_dir)
    dfs = []
    for ship_type in SHIP_TYPES:
        type_dir = train_dir / ship_type
        if not type_dir.exists():
            logging.warning(f"目录 {type_dir} 不存在，跳过")
            continue
        for filename in tqdm(list(type_dir.glob("*.csv")), desc=f"加载{ship_type}数据"):
            try:
                df = pd.read_csv(filename, dtype={"shipno": str})
                df["shiptype"] = ship_type
                df = df[[col for col in df.columns if col in TRAJECTORY_COLS + ["shiptype"]]]
                dfs.append(df)
            except Exception as e:
                logging.error(f"加载文件 {filename} 失败: {e}")
    if not dfs:
        raise FileNotFoundError(f"未在 {train_dir} 找到训练 CSV")
    train_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"训练数据列类型:\n{train_df.dtypes}")
    return train_df


def load_train_data_split(train_split_dir, num_splits=8) -> pd.DataFrame:
    cfg = Config(num_splits=num_splits)
    train_split_dir = Path(train_split_dir)
    dfs = []

    for split_type in SPLIT_TYPES:
        split_dir = train_split_dir / split_type
        if not split_dir.exists():
            logging.warning(f"目录 {split_dir} 不存在，跳过")
            continue

        for i in range(1, cfg.max_split_id(split_type) + 1):
            month_dir = split_dir / f"month_{i}"
            if not month_dir.exists():
                logging.warning(f"目录 {month_dir} 不存在，跳过")
                continue
            for ship_type in SHIP_TYPES:
                type_dir = month_dir / ship_type
                if not type_dir.exists():
                    logging.warning(f"目录 {type_dir} 不存在，跳过")
                    continue
                files = list(type_dir.glob("*.csv"))
                for filename in tqdm(files, desc=f"加载 {split_type}/month_{i}/{ship_type} 数据"):
                    try:
                        df = pd.read_csv(filename, dtype={"shipno": str})
                        df["shiptype"] = ship_type
                        df["month_split"] = i
                        df["split_type"] = split_type
                        valid_columns = TRAJECTORY_COLS + ["shiptype", "month_split", "split_type"]
                        df = df[[col for col in df.columns if col in valid_columns]]
                        dfs.append(df)
                    except Exception as e:
                        logging.error(f"加载文件 {filename} 失败: {e}")

    if not dfs:
        raise FileNotFoundError(f"未在 {train_split_dir} 找到切分后的训练数据，请先运行 split")
    train_df = pd.concat(dfs, ignore_index=True)
    train_df["postime"] = parse_postime(train_df["postime"])
    train_df = train_df.dropna(subset=["postime"])
    logging.info(f"训练数据列类型:\n{train_df.dtypes}")
    logging.info(
        f"总训练数据量: {len(train_df)} 条，船舶数: "
        f"{train_df[['shipno', 'month_split', 'split_type']].drop_duplicates().shape[0]}"
    )
    return train_df


def load_test_data(test_dir) -> pd.DataFrame:
    test_dir = Path(test_dir)
    dfs = []
    required_columns = ["shipno", "postime", "lon", "lat", "sog"]
    for filename in tqdm(list(test_dir.glob("*.csv")), desc="加载测试数据"):
        try:
            df = pd.read_csv(filename, dtype={"shipno": str})
            if not all(col in df.columns for col in required_columns):
                logging.error(f"文件 {filename} 缺少必要列: {set(required_columns) - set(df.columns)}")
                continue
            df = df[[col for col in df.columns if col in TRAJECTORY_COLS]]
            dfs.append(df)
        except Exception as e:
            logging.error(f"加载文件 {filename} 失败: {e}")
    if not dfs:
        raise FileNotFoundError(f"未在 {test_dir} 找到测试 CSV")
    test_df = pd.concat(dfs, ignore_index=True)
    test_df["postime"] = parse_postime(test_df["postime"])
    test_df = test_df.dropna(subset=["postime"])
    logging.info(f"测试数据列类型:\n{test_df.dtypes}")
    logging.info(f"测试数据量: {len(test_df)} 条，船舶数: {test_df['shipno'].nunique()}")
    return test_df
