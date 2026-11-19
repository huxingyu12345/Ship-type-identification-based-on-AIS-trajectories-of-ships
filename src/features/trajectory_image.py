import logging
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config import SPLIT_TYPES, Config
from src.data.preprocess import denoise_trajectory


def generate_trajectory_image(group, output_path, img_size=(224, 224)):
    output_path = str(output_path)
    if len(group) < 2:
        img = Image.new("RGB", img_size, color="white")
        img.save(output_path)
        if len(group) == 0:
            logging.warning(f"Empty group for ship, generated blank image at {output_path}")
        else:
            logging.warning(f"Ship {group['shipno'].iloc[0]} has fewer than 2 points, generated blank image.")
        return output_path

    group = group.sort_values("postime")
    lat, lon, sog, postime = group["lat"], group["lon"], group["sog"], group["postime"]

    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    if lat_max == lat_min:
        lat_max += 1e-6
    if lon_max == lon_min:
        lon_max += 1e-6
    sog_min, sog_max = sog.min(), sog.max()
    if sog_max == sog_min:
        sog_max += 1e-6

    img = Image.new("RGB", img_size, color="white")
    draw = ImageDraw.Draw(img)

    points = []
    time_min = postime.min().timestamp()
    time_max = postime.max().timestamp()
    for i in range(len(group)):
        x = int((lon.iloc[i] - lon_min) / (lon_max - lon_min) * (img_size[0] - 1))
        y = int((1 - (lat.iloc[i] - lat_min) / (lat_max - lat_min)) * (img_size[1] - 1))
        points.append((x, y))

    for i in range(1, len(points)):
        normalized_time = (postime.iloc[i].timestamp() - time_min) / (time_max - time_min + 1e-10)
        color = (
            int(255 * normalized_time),
            int(255 * (1 - normalized_time)),
            int(255 * (sog.iloc[i] - sog_min) / (sog_max - sog_min + 1e-10)),
        )
        width = int(1 + 4 * (sog.iloc[i] - sog_min) / (sog_max - sog_min + 1e-10))
        draw.line([points[i - 1], points[i]], fill=color, width=width)

    img.save(output_path)
    return output_path


def process_ship(args):
    shipno, group, output_dir, is_train, img_size = args
    img_path = os.path.join(output_dir, f"{shipno}.png")

    try:
        if len(group) == 0:
            logging.warning(f"Ship {shipno} has no data points, skipping image generation.")
            return None

        if not os.path.exists(img_path):
            group = denoise_trajectory(group)
            generate_trajectory_image(group, img_path, img_size=img_size)

        shiptype = None
        if is_train and "shiptype" in group.columns and not group.empty:
            try:
                shiptype = group["shiptype"].iloc[0]
            except IndexError:
                logging.warning(f"Ship {shipno} has empty shiptype after processing, setting shiptype to None.")

        return {
            "shipno": shipno,
            "img_path": img_path,
            "shiptype": shiptype,
            "point_count": len(group),
        }
    except Exception as e:
        logging.error(f"处理船舶 {shipno} 失败: {str(e)}")
        return None


def generate_images_parallel(df, output_dir, is_train=True, n_jobs=None, img_size=(224, 224)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or Config().n_jobs

    grouped = list(df.groupby("shipno"))
    args = [(shipno, group, str(output_dir), is_train, img_size) for shipno, group in grouped]

    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(process_ship, args), total=len(args), desc="生成图像"))

    results = [r for r in results if r is not None]
    if not results:
        logging.warning("未生成任何图像，创建空的 meta DataFrame")
        return pd.DataFrame(columns=["shipno", "img_path", "shiptype", "point_count"])
    return pd.DataFrame(results)


def generate_images_parallel_split(train_df, train_images_base_dir, num_splits=8, is_train=True, n_jobs=None):
    cfg = Config(num_splits=num_splits)
    train_images_base_dir = Path(train_images_base_dir)
    train_images_base_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or cfg.n_jobs

    image_meta_list = []
    total_images = 0

    for split_type in SPLIT_TYPES:
        split_dir = train_images_base_dir / split_type
        split_df = train_df[train_df["split_type"] == split_type]
        if len(split_df) == 0:
            logging.warning(f"划分 {split_type} 无数据，跳过图像生成")
            continue

        for i in range(1, cfg.max_split_id(split_type) + 1):
            month_dir = split_dir / f"month_{i}"
            month_df = split_df[split_df["month_split"] == i]
            if len(month_df) == 0:
                logging.warning(f"划分 {split_type}/month_{i} 无数据，跳过图像生成")
                continue

            logging.info(f"划分 {split_type}/month_{i} 包含 {month_df['shipno'].nunique()} 艘船舶")
            month_meta = generate_images_parallel(
                month_df, month_dir, is_train=is_train, n_jobs=n_jobs, img_size=cfg.img_size
            )
            if not month_meta.empty:
                month_meta["month_split"] = i
                month_meta["split_type"] = split_type
                month_dir.mkdir(parents=True, exist_ok=True)
                month_meta.to_csv(month_dir / "meta.csv", index=False)
                image_meta_list.append(month_meta)
                total_images += len(month_meta)
                logging.info(f"划分 {split_type}/month_{i}: 生成 {len(month_meta)} 张图像")
            else:
                logging.warning(f"划分 {split_type}/month_{i}: 未生成任何图像")

    if image_meta_list:
        image_meta = pd.concat(image_meta_list, ignore_index=True)
        image_meta.to_csv(train_images_base_dir / "meta.csv", index=False)
        logging.info(f"总计生成 {len(image_meta)} 张图像（{total_images} 张图像累计）")
    else:
        image_meta = pd.DataFrame(
            columns=["shipno", "img_path", "shiptype", "month_split", "split_type", "point_count"]
        )
        logging.warning("未生成任何图像，创建空的 image_meta")
    return image_meta
