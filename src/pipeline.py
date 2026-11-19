from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ID_COLS, Config
from src.data.preprocess import split_exists
from src.data.splitter import run_split
from src.utils.common import clean_shipno, setup_logging


def _load_or_extract_features(path: Path, extract_fn):
    features = None
    if path.exists():
        print(f"✅ 读取缓存特征: {path}")
        features = pd.read_csv(path)
        non_numeric_cols = features.select_dtypes(exclude=[np.number, "object"]).columns
        if non_numeric_cols.any() or "code_hash" in features.columns:
            logging.warning(f"缓存文件 {path} 格式不正确，重新提取特征")
            path.unlink()
            features = None
    if features is None:
        print(f"🚀 正在提取特征 -> {path}")
        features = extract_fn().fillna(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(path, index=False)
        print(f"✅ 特征保存至 {path}")
        logging.info(f"特征样本数: {len(features)}")
    return features


def _align_train(train_features, train_image_meta):
    print("🚀 对齐训练集特征和图像元数据")
    keys = ["shipno", "month_split", "split_type"]
    train_keys = train_features[keys].apply(tuple, axis=1)
    image_keys = train_image_meta[keys].apply(tuple, axis=1)
    train_features = train_features[train_keys.isin(image_keys)]
    train_image_meta = train_image_meta[image_keys.isin(train_keys)]
    train_features = train_features.sort_values(keys)
    train_image_meta = train_image_meta.sort_values(keys)
    logging.info(
        f"对齐后训练特征样本数: {len(train_features)}, 图像元数据样本数: {len(train_image_meta)}"
    )
    if len(train_features) != len(train_image_meta):
        raise ValueError("训练特征和图像元数据样本数不一致")
    return train_features, train_image_meta


def _align_test(test_features, test_image_meta):
    print("🚀 对齐测试集特征和图像元数据")
    test_features = test_features[test_features["shipno"].isin(test_image_meta["shipno"])]
    test_image_meta = test_image_meta[test_image_meta["shipno"].isin(test_features["shipno"])]
    test_features = test_features.sort_values("shipno")
    test_image_meta = test_image_meta.sort_values("shipno")
    logging.info(
        f"对齐后测试特征样本数: {len(test_features)}, 测试图像元数据样本数: {len(test_image_meta)}"
    )
    if len(test_features) != len(test_image_meta):
        raise ValueError("测试特征和图像元数据样本数不一致")
    return test_features, test_image_meta


def run_train(cfg: Optional[Config] = None):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping

    from src.data.loader import load_test_data, load_train_data_split
    from src.features.cnn import extract_image_features
    from src.features.tabular import extract_features_parallel
    from src.features.trajectory_image import generate_images_parallel, generate_images_parallel_split
    from src.models.hybrid import CosineAnnealingScheduler, build_hybrid_model, create_dataset
    from src.models.selection import select_image_features, select_tabular_features

    cfg = cfg or Config()
    cfg.ensure_dirs()
    setup_logging(cfg.train_log_path)

    tf.keras.mixed_precision.set_global_policy("float32")
    logging.info("混合精度策略设置为 float32")

    def extract_train():
        train_raw = load_train_data_split(cfg.train_split_dir, num_splits=cfg.num_splits)
        return extract_features_parallel(train_raw, is_train=True, n_jobs=cfg.n_jobs)

    def extract_test():
        test_raw = load_test_data(cfg.test_dir)
        return extract_features_parallel(test_raw, is_train=False, n_jobs=cfg.n_jobs)

    train_features = _load_or_extract_features(cfg.train_features_path, extract_train)
    if "split_type" not in train_features.columns:
        logging.warning("训练特征缺少 split_type，重新提取")
        cfg.train_features_path.unlink(missing_ok=True)
        train_features = _load_or_extract_features(cfg.train_features_path, extract_train)

    test_features = _load_or_extract_features(cfg.test_features_path, extract_test)

    train_meta_path = cfg.train_images_dir / "meta.csv"
    if train_meta_path.exists():
        print("✅ 读取缓存的训练图像元数据")
        train_image_meta = pd.read_csv(train_meta_path)
    else:
        print("🚀 正在生成训练图像")
        train_raw = load_train_data_split(cfg.train_split_dir, num_splits=cfg.num_splits)
        train_image_meta = generate_images_parallel_split(
            train_raw,
            cfg.train_images_dir,
            num_splits=cfg.num_splits,
            is_train=True,
            n_jobs=cfg.n_jobs,
        )
        print(f"✅ 训练图像元数据保存至 {train_meta_path}")

    test_meta_path = cfg.test_images_dir / "meta.csv"
    if test_meta_path.exists():
        print("✅ 读取缓存的测试图像元数据")
        test_image_meta = pd.read_csv(test_meta_path)
    else:
        print("🚀 正在生成测试图像")
        test_raw = load_test_data(cfg.test_dir)
        test_image_meta = generate_images_parallel(
            test_raw, cfg.test_images_dir, is_train=False, n_jobs=cfg.n_jobs, img_size=cfg.img_size
        )
        test_image_meta.to_csv(test_meta_path, index=False)
        print(f"✅ 测试图像元数据保存至 {test_meta_path}")

    train_features, train_image_meta = _align_train(train_features, train_image_meta)
    test_features, test_image_meta = _align_test(test_features, test_image_meta)

    le = LabelEncoder()
    y = le.fit_transform(train_features["shiptype"])
    logging.info(f"标签 y 样本数: {len(y)}")

    drop_cols = [col for col in ID_COLS if col in train_features.columns]
    feature_cols = train_features.drop(drop_cols, axis=1, errors="ignore").select_dtypes(include=[np.number]).columns
    X_tabular = (
        train_features.drop(drop_cols, axis=1, errors="ignore")
        .select_dtypes(include=[np.number])
        .fillna(0)
        .astype(np.float32)
    )
    test_drop_cols = [col for col in ["shipno", "shiptype", "month_split", "split_type"] if col in test_features.columns]
    X_test_tabular = (
        test_features.drop(test_drop_cols, axis=1, errors="ignore")
        .select_dtypes(include=[np.number])
        .fillna(0)
        .astype(np.float32)
    )

    scaler = StandardScaler()
    X_tabular = scaler.fit_transform(X_tabular)
    X_test_tabular = scaler.transform(X_test_tabular)

    print("🚀 提取训练集图像特征")
    train_image_features = extract_image_features(
        train_image_meta["img_path"], batch_size=cfg.image_batch_size, img_size=cfg.img_size
    )
    print(f"训练集图像特征样本数: {len(train_image_features)}")
    print("🚀 提取测试集图像特征")
    test_image_features = extract_image_features(
        test_image_meta["img_path"], batch_size=cfg.image_batch_size, img_size=cfg.img_size
    )
    print(f"测试集图像特征样本数: {len(test_image_features)}")

    if len(X_tabular) != len(train_image_features) or len(X_tabular) != len(y):
        raise ValueError("训练集特征和标签样本数不一致")
    if len(X_test_tabular) != len(test_image_features):
        raise ValueError("测试集特征样本数不一致")

    X_tabular, X_test_tabular, _ = select_tabular_features(
        X_tabular, y, feature_cols, X_test_tabular, cfg
    )
    train_image_features, test_image_features, _ = select_image_features(
        train_image_features, test_image_features, y, cfg
    )

    print("===== 添加特征交互 =====")
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_tabular = poly.fit_transform(X_tabular)
    X_test_tabular = poly.transform(X_test_tabular)

    print("===== 合并表格和图像特征 =====")
    X_all = np.concatenate([X_tabular, train_image_features], axis=1)
    X_test_all = np.concatenate([X_test_tabular, test_image_features], axis=1)
    print(f"合并后训练特征维度: {X_all.shape}")
    print(f"合并后测试特征维度: {X_test_all.shape}")

    print("===== 计算类权重 =====")
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weights_dict = dict(zip(le.classes_, class_weights))
    class_weights_array = np.array(
        [class_weights_dict[le.classes_[i]] for i in range(len(le.classes_))], dtype=np.float32
    )
    sample_weights = np.array(
        [
            class_weights_dict[le.classes_[label]]
            * (
                1.0
                if split_type == "1month"
                else 1.2
                if split_type == "2month"
                else 1.4
                if split_type == "4month"
                else 1.6
            )
            for label, split_type in zip(y, train_image_meta["split_type"])
        ],
        dtype=np.float32,
    )
    print("类权重:", class_weights_dict)
    logging.info(f"类权重: {class_weights_dict}")

    print("===== 交叉验证训练神经网络模型 =====")
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    val_accuracies = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
        print(f"🚀 训练第 {fold + 1} 折")
        y_train, y_val = y[train_idx], y[val_idx]
        X_train_tabular, X_val_tabular = X_tabular[train_idx], X_tabular[val_idx]
        X_train_image, X_val_image = train_image_features[train_idx], train_image_features[val_idx]
        train_sample_weights = sample_weights[train_idx]
        val_sample_weights = sample_weights[val_idx]

        train_dataset = create_dataset(
            X_train_tabular, X_train_image, y_train, train_sample_weights, batch_size=cfg.batch_size
        )
        val_dataset = create_dataset(
            X_val_tabular, X_val_image, y_val, val_sample_weights, batch_size=cfg.batch_size, shuffle=False
        )

        model = build_hybrid_model(
            tabular_input_shape=X_tabular.shape[1],
            image_feature_shape=train_image_features.shape[1],
            num_classes=len(le.classes_),
            class_weights=class_weights_array,
        )
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=cfg.epochs,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=cfg.early_stopping_patience, restore_best_weights=True),
                CosineAnnealingScheduler(eta_max=0.001, eta_min=1e-5, T_max=10),
            ],
            verbose=1,
        )
        _ = history

        val_pred_labels = np.argmax(model.predict(val_dataset), axis=1)
        val_accuracy = accuracy_score(y_val, val_pred_labels)
        val_accuracies.append(val_accuracy)
        print(f"第 {fold + 1} 折验证准确率: {val_accuracy:.4f}")
        logging.info(f"第 {fold + 1} 折验证准确率: {val_accuracy:.4f}")
        models.append(model)

    print(f"交叉验证平均准确率: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    logging.info(f"交叉验证平均准确率: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")

    print("===== 使用最佳模型进行测试集预测 =====")
    best_model = models[np.argmax(val_accuracies)]
    test_dataset = create_dataset(
        X_test_tabular,
        test_image_features,
        np.zeros(len(X_test_tabular)),
        np.ones(len(X_test_tabular)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_pred_labels = le.inverse_transform(np.argmax(best_model.predict(test_dataset), axis=1))
    pred_counts = pd.Series(test_pred_labels).value_counts()
    print("测试集预测分布:\n", pred_counts)
    logging.info(f"测试集预测分布:\n{pred_counts}")

    result = pd.DataFrame(
        {
            "shipno": clean_shipno(test_image_meta["shipno"]),
            "shiptype": test_pred_labels,
        }
    )
    result.to_csv(cfg.result_path, index=False, encoding="utf-8")
    print("✅ 提交文件已保存:", cfg.result_path)
    return result


def run_pipeline(step: str = "all", force_split: bool = False, num_splits: Optional[int] = None):
    cfg = Config()
    if num_splits is not None:
        cfg.num_splits = num_splits
    cfg.ensure_dirs()

    if step in ("split", "all"):
        if step == "split" or force_split or not split_exists(cfg):
            run_split(cfg)
        else:
            print(f"检测到已有切分数据: {cfg.train_split_dir}，跳过 split。使用 --force-split 强制重切。")

    if step in ("train", "all"):
        if not split_exists(cfg):
            raise FileNotFoundError(
                f"未找到切分数据 {cfg.train_split_dir}，请先运行: python main.py --step split"
            )
        run_train(cfg)
