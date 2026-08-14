"""Unit tests for src.augmentation - decoding and augmentation, in isolation
from the rest of the tf.data pipeline."""

import numpy as np
import tensorflow as tf
from PIL import Image

from src.augmentation.augment import augment_image, decode_and_resize


def test_decode_and_resize_returns_requested_size(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 30), color=(10, 20, 30)).save(img_path)

    resized = decode_and_resize(tf.constant(str(img_path)), (224, 224))

    assert resized.shape == (224, 224, 3)


def test_augment_image_preserves_shape_and_dtype():
    image = tf.random.uniform((224, 224, 3), maxval=255, dtype=tf.float32)
    augmented = augment_image(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype


def test_augment_image_is_stochastic_across_calls():
    # With random flip/rotation/zoom/contrast all enabled, two calls on the
    # same input should not be identical (guards against an accidentally
    # no-op augmentation pipeline).
    image = tf.constant(np.random.rand(224, 224, 3).astype("float32") * 255)
    out1 = augment_image(image).numpy()
    out2 = augment_image(image).numpy()
    assert not np.array_equal(out1, out2)
