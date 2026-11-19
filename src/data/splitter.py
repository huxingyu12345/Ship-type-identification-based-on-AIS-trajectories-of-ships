from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import SPLIT_TYPES, Config
from src.data.loader import load_train_data
from src.utils.common import clean_shipno, parse_postime, setup_logging


def split_train_data_by_month(train_df, output_base_dir, num_splits=8):
    output_base_dir = Path(output_base_dir)
    train_df = train_df.copy()
    train_df["postime"] = parse_postime(train_df["postime"])
    train_df = train_df.dropna(subset=["postime"])
    logging.info(f"有效时间戳数据量: {len(train_df)}")

    min_time = train_df["postime"].min()
    max_time = train_df["postime"].max()
    total_duration = (max_time - min_time).total_seconds()
    split_duration = total_duration / num_splits
    logging.info(
        f"数据时间范围: {min_time} 至 {max_time}, 每1个月约 {split_duration / (24 * 3600)} 天"
    )

    output_base_dir.mkdir(parents=True, exist_ok=True)
    split_dfs = []

    for split_type in SPLIT_TYPES:
        if split_type == "1month":
            n_splits = num_splits
            duration = split_duration
        elif split_type == "2month":
            n_splits = num_splits // 2
            duration = split_duration * 2
        elif split_type == "4month":
            n_splits = num_splits // 4
            duration = split_duration * 4
        else:
            n_splits = 1
            duration = total_duration

        split_dir = output_base_dir / split_type
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_splits):
            month_dir = split_dir / f"month_{i + 1}"
            month_dir.mkdir(parents=True, exist_ok=True)

            start_time = min_time + timedelta(seconds=i * duration)
            end_time = (
                min_time + timedelta(seconds=(i + 1) * duration)
                if i < n_splits - 1
                else max_time + timedelta(seconds=1)
            )

            mask = (train_df["postime"] >= start_time) & (train_df["postime"] < end_time)
            split_df = train_df[mask].copy()
            split_df["month_split"] = i + 1
            split_df["split_type"] = split_type
            split_dfs.append(split_df)
            logging.info(
                f"划分 {split_type}/month_{i + 1} ({start_time} 至 {end_time}): {len(split_df)} 条记录"
            )

            for ship_type in split_df["shiptype"].unique():
                type_dir = month_dir / ship_type
                type_dir.mkdir(parents=True, exist_ok=True)
                type_df = split_df[split_df["shiptype"] == ship_type]
                for shipno, group in tqdm(
                    type_df.groupby("shipno"),
                    desc=f"保存 {split_type}/month_{i + 1}/{ship_type}",
                ):
                    output_path = type_dir / f"{shipno}.csv"
                    group = group.drop(columns=["month_split", "split_type"], errors="ignore")
                    group.to_csv(output_path, index=False, encoding="utf-8")
                    logging.info(f"保存 {output_path}, 记录数: {len(group)}")

    combined_df = pd.concat(split_dfs, ignore_index=True)
    logging.info(
        f"合并后总数据量: {len(combined_df)} 条，唯一船舶数: "
        f"{combined_df[['shipno', 'month_split', 'split_type']].drop_duplicates().shape[0]}"
    )
    return combined_df


def _print_split_stats(combined_df, num_splits: int) -> None:
    cfg = Config(num_splits=num_splits)
    for split_type in SPLIT_TYPES:
        split_df = combined_df[combined_df["split_type"] == split_type]
        for i in range(1, cfg.max_split_id(split_type) + 1):
            month_df = split_df[split_df["month_split"] == i]
            if len(month_df) == 0:
                continue
            print(
                f"划分 {split_type}/month_{i}: {len(month_df)} 条记录，船舶数: {month_df['shipno'].nunique()}"
            )
            print(f"船型分布:\n{month_df['shiptype'].value_counts()}")
            logging.info(
                f"划分 {split_type}/month_{i}: {len(month_df)} 条记录，船舶数: {month_df['shipno'].nunique()}"
            )
            logging.info(f"船型分布:\n{month_df['shiptype'].value_counts()}")


def run_split(cfg: Optional[Config] = None):
    cfg = cfg or Config()
    cfg.ensure_dirs()
    setup_logging(cfg.split_log_path)

    print("🚀 加载训练数据...")
    train_df = load_train_data(cfg.train_dir)
    train_df["shipno"] = clean_shipno(train_df["shipno"])
    logging.info(f"总训练数据量: {len(train_df)} 条，船舶数: {train_df['shipno'].nunique()}")

    print("🚀 开始按时间分割训练数据...")
    combined_df = split_train_data_by_month(
        train_df, cfg.train_split_dir, num_splits=cfg.num_splits
    )
    _print_split_stats(combined_df, cfg.num_splits)
    print(f"✅ 切分完成，输出目录: {cfg.train_split_dir}")
    return combined_df
