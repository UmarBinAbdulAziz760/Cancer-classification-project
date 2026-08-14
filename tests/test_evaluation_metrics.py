"""Unit tests for src.evaluation.metrics, using a hand-computable toy example
instead of a real trained model."""

import numpy as np
import tensorflow as tf

from src.evaluation.metrics import evaluate_model


class _StubModel:
    """Returns fixed, pre-baked softmax probabilities regardless of input,
    so the resulting metrics can be checked by hand."""

    def __init__(self, probs_by_batch):
        self._probs_by_batch = probs_by_batch
        self._call = 0

    def predict(self, images, verbose=0):
        probs = self._probs_by_batch[self._call]
        self._call += 1
        return probs


def test_evaluate_model_binary_perfect_predictions():
    # 4 examples, 2 classes, model predicts perfectly.
    images = tf.zeros((4, 2, 2, 3))
    labels = tf.constant([0, 0, 1, 1])
    test_ds = [(images, labels)]

    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    model = _StubModel([probs])

    metrics = evaluate_model(model, test_ds, num_classes=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["auc_roc"] == 1.0
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_evaluate_model_binary_with_one_mistake():
    images = tf.zeros((4, 2, 2, 3))
    labels = tf.constant([0, 0, 1, 1])
    test_ds = [(images, labels)]

    # last example misclassified as class 0
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.6, 0.4],
    ])
    model = _StubModel([probs])

    metrics = evaluate_model(model, test_ds, num_classes=2)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [1, 1]]


def test_evaluate_model_multiclass_returns_macro_auc():
    images = tf.zeros((3, 2, 2, 3))
    labels = tf.constant([0, 1, 2])
    test_ds = [(images, labels)]

    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    model = _StubModel([probs])

    metrics = evaluate_model(model, test_ds, num_classes=3)

    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)
