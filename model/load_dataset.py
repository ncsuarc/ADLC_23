"""
https://keras.io/guides/keras_cv/object_detection_keras_cv/
"""

# import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from adlc_util import visualize_dataset

BATCH_SIZE = 4
RAND_SEED = 0  # None for final


def read_annotations(file="data/annotation_238.txt", test_size=0.2):
    images = []
    boxes = []
    classes = []
    full_ds_size = 0
    with open(file) as f:
        for line in f.readlines():
            # Split line on whitespace
            line = line.split()

            # If no box for given image, skip
            box_count = int(line[1])
            if box_count == 0:
                continue

            full_ds_size += 1
            # Load image and convert
            images += [tf.convert_to_tensor(Image.open(f"./data/{line[0]}"))]

            current_box = []
            current_class = []
            for i in range(box_count):
                # Groups of 4 as x,y,w,h values
                current_box += [[int(x) for x in line[2 + i * 4 : 2 + i * 4 + 4]]]
                current_class += [0]

            boxes += [current_box]
            classes += [current_class]

    data = {
        "images": images,
        "bounding_boxes": {
            "boxes": tf.ragged.constant(boxes).to_tensor(),
            "classes": tf.ragged.constant(classes).to_tensor(),
        },
    }

    full_ds = tf.data.Dataset.from_tensor_slices(data)

    # train/test split (somewhat flawed b/c needs size of ds)
    # https://stackoverflow.com/a/60894496/6440256d
    full_ds.shuffle(full_ds_size, seed=RAND_SEED)

    train_ds_size = int(0.8 * full_ds_size)
    train_ds = full_ds.take(train_ds_size)
    test_ds = full_ds.skip(train_ds_size)

    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = read_annotations()

    # Batch observations
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    test_ds = test_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

    # Persist datasets
    train_ds.save("data/238_train_ds", compression="GZIP")
    test_ds.save("data/238_test_ds", compression="GZIP")

    # print(train_ds)

    visualize_dataset(
        train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=BATCH_SIZE, cols=1
    )
