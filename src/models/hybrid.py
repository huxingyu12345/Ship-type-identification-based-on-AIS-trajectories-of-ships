import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, eta_max=0.001, eta_min=1e-5, T_max=10):
        super().__init__()
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = T_max

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(np.pi * epoch / self.T_max))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.cast(tf.gather(class_weights, y_true), tf.float32)
        unweighted_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        unweighted_losses = tf.cast(unweighted_losses, tf.float32)
        weighted_losses = unweighted_losses * weights
        return tf.reduce_mean(weighted_losses)

    return loss


def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    se = Dense(channels // reduction_ratio, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    return tf.keras.layers.Multiply()([input_tensor, se])


def build_hybrid_model(tabular_input_shape, image_feature_shape, num_classes, class_weights):
    tabular_input = Input(shape=(tabular_input_shape,), dtype=tf.float32, name="tabular_input")
    tabular = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(tabular_input)
    tabular = BatchNormalization()(tabular)
    tabular = Dropout(0.15)(tabular)
    tabular = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(tabular)
    tabular = BatchNormalization()(tabular)
    tabular = Dropout(0.15)(tabular)
    tabular = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(tabular)
    tabular = LayerNormalization()(tabular)
    tabular = se_block(tf.expand_dims(tabular, axis=1))
    tabular = tf.squeeze(tabular, axis=1)

    image_input = Input(shape=(image_feature_shape,), dtype=tf.float32, name="image_input")
    image = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(image_input)
    image = BatchNormalization()(image)
    image = Dropout(0.15)(image)
    image = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(image)
    image = BatchNormalization()(image)
    image = Dropout(0.15)(image)
    image = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(image)
    image = LayerNormalization()(image)
    image = se_block(tf.expand_dims(image, axis=1))
    image = tf.squeeze(image, axis=1)

    combined = Concatenate()([tabular, image])
    combined = tf.expand_dims(combined, axis=1)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(combined, combined)
    attention = tf.squeeze(attention, axis=1)

    residual = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(attention)
    combined = Add()([attention, residual])
    combined = LayerNormalization()(combined)
    combined = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.15)(combined)

    output = Dense(num_classes, activation="softmax", name="output")(combined)
    model = Model(inputs=[tabular_input, image_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
        loss=weighted_categorical_crossentropy(class_weights),
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )
    return model


def create_dataset(tabular, image, labels, sample_weights, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"tabular_input": tabular, "image_input": image}, labels, sample_weights)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
