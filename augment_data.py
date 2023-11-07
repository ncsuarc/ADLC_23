import keras_cv
import tensorflow as tf
from adlc_util import visualize_dataset


def get_augmenter():
    return tf.keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format="xywh",
            ),
        ]
    )


if __name__ == "__main__":
    # NOTE: this requires tensorflow>=2.14 on linux: https://github.com/XiaotingChen/maxatac_pip_1.0.5/issues/2
    train_ds = tf.data.Dataset.load(path="data/238_train_ds", compression="GZIP")
    test_ds = tf.data.Dataset.load(path="data/238_test_ds", compression="GZIP")

    augmenter = get_augmenter()

    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    visualize_dataset(
        train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2, offset=2
    )
