from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from src.config import Config


def build_xgb_classifier(num_class, cfg: Optional[Config] = None):
    cfg = cfg or Config()
    params = {
        "n_estimators": cfg.xgb_n_estimators,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "objective": "multi:softmax",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": cfg.xgb_n_jobs,
    }
    if cfg.xgb_tree_method:
        params["tree_method"] = cfg.xgb_tree_method
        if cfg.xgb_tree_method == "gpu_hist":
            params["gpu_id"] = cfg.xgb_gpu_id
    return xgb.XGBClassifier(**params)


def select_tabular_features(X_tabular, y, feature_cols, X_test_tabular, cfg: Config):
    print("===== 使用 RFE 进行表格特征选择 =====")
    model = build_xgb_classifier(num_class=len(set(y)), cfg=cfg)
    rfe = RFE(estimator=model, n_features_to_select=cfg.rfe_n_features)
    rfe.fit(X_tabular, y)
    selected_features = feature_cols[rfe.support_]
    print(f"选择后的表格特征: {list(selected_features)}")
    logging.info(f"选择后的表格特征: {list(selected_features)}")
    return X_tabular[:, rfe.support_], X_test_tabular[:, rfe.support_], selected_features


def select_image_features(train_image_features, test_image_features, y, cfg: Config):
    print("===== 进行图像特征选择 =====")
    model = build_xgb_classifier(num_class=len(set(y)), cfg=cfg)
    X_train_image, _, y_train, _ = train_test_split(
        train_image_features, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train_image, y_train)

    image_feature_cols = [f"image_feature_{i}" for i in range(train_image_features.shape[1])]
    feature_importance_image = pd.DataFrame(
        {"feature": image_feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("图像特征重要性:\n", feature_importance_image)
    logging.info(f"图像特征重要性:\n{feature_importance_image}")

    top_indices = feature_importance_image.index[: cfg.image_top_k].values
    print(f"选择后的图像特征数量: {len(top_indices)}")
    print(f"选择的图像特征索引: {list(top_indices)}")
    logging.info(f"选择的图像特征索引: {list(top_indices)}")
    return train_image_features[:, top_indices], test_image_features[:, top_indices], top_indices
