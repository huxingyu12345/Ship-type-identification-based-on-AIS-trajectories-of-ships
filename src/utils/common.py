import logging
from pathlib import Path

import pandas as pd


def clean_shipno(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"e\+\d+", "", regex=True)
    )


def parse_postime(series: pd.Series) -> pd.Series:
    """将 AIS 时间戳解析为 datetime。原始数据为 Unix 秒。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().median()
        if median > 1e12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def setup_logging(log_path, also_console: bool = True) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if also_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
