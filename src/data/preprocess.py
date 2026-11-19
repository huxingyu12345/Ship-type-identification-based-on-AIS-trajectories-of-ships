import logging

from src.config import Config


def denoise_trajectory(group):
    original_len = len(group)
    group = group[(group["lat"].between(-90, 90)) & (group["lon"].between(-180, 180))]
    group = group[group["sog"].between(0, 50)]
    group["sog"] = group["sog"].rolling(window=3, min_periods=1, center=True).mean()
    group["cog"] = group["cog"].rolling(window=3, min_periods=1, center=True).mean()
    group = group.dropna()
    if len(group) < original_len:
        shipno = group["shipno"].iloc[0] if len(group) > 0 else "unknown"
        logging.info(f"Ship {shipno}: {original_len - len(group)} points filtered out during denoising.")
    return group


def segment_trajectory(group, points_per_segment=100):
    segments = []
    group = group.sort_values("postime")
    total_points = len(group)
    for i in range(0, total_points, points_per_segment):
        segment = group.iloc[i : i + points_per_segment]
        if len(segment) >= 2:
            segments.append(segment)
    return segments


def split_exists(cfg: Config) -> bool:
    return (cfg.train_split_dir / "1month" / "month_1").exists()
