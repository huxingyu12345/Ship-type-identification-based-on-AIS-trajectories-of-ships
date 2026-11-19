import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


def extract_image_features(img_paths, batch_size=16, img_size=(224, 224)):
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    features = []
    img_paths = list(img_paths)
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = datagen.random_transform(img)
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                batch_images.append(img)
            except Exception as e:
                logging.warning(f"无法加载图像 {path}: {e}")
                batch_images.append(np.zeros((*img_size, 3)))
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_features = model.predict(batch_images, verbose=0)
        features.extend(batch_features)
    return np.array(features, dtype=np.float32)
