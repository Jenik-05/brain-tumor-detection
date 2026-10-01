"""Reads image files and turns them into model-ready batches (tf.data)."""
import tensorflow as tf

from src import config


def load_and_preprocess(filepath, label):
    """One image file -> (224x224x3 float image with values 0-255, label)."""
    file_bytes = tf.io.read_file(filepath)
    # channels=1 -> grayscale. Handles RGB, grayscale, RGBA and palette images,
    # so colour-format differences between classes disappear.
    image = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    image = tf.image.resize(image, config.IMAGE_SIZE, antialias=True)
    # Pretrained models expect 3 channels: repeat the gray channel 3 times.
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def make_dataset(df, training=False):
    """Turn a table (filepath + label columns) into a batched tf.data pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values)
    )
    if training:
        dataset = dataset.shuffle(len(df), seed=config.SEED)   # new order each epoch
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)                   # prepare next batch early