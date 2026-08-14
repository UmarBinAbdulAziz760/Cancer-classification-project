"""
Training-time image augmentation and raw-file decoding.

Applied to the training split only (Section III-A): random flip, small
rotation, zoom, and contrast jitter. Validation and test splits skip this
module entirely and go straight from decode -> architecture preprocessing.
"""

import tensorflow as tf

_augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="augmentation_pipeline")


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Applies the augmentation pipeline to a single (H, W, 3) image tensor."""
    image = tf.expand_dims(image, axis=0)
    image = _augmentation_pipeline(image, training=True)
    return tf.squeeze(image, axis=0)


def decode_and_resize(filepath: tf.Tensor, target_size) -> tf.Tensor:
    """Reads an image file from disk and resizes it to `target_size` (H, W)."""
    raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, target_size)
