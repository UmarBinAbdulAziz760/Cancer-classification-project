"""
Transfer-learning model construction shared by all four architectures.

All four architectures share an identical classifier head - global average
pooling, a dense layer, dropout, and a softmax output - so that comparisons
reflect the pretrained feature extractors rather than head design
(Section III-B).
"""

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, DenseNet121
import tensorflow as tf

from .. import config

BASE_CONSTRUCTORS = {
    "vgg16": VGG16,
    "resnet50": ResNet50,
    "mobilenet": MobileNet,
    "densenet121": DenseNet121,
}


def build_transfer_model(
    architecture: str,
    input_shape,
    num_classes: int,
    dense_units: int = 256,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-3,
):
    """Builds a transfer-learning model using locally stored ImageNet weights
    (config.WEIGHTS_PATHS), with the backbone frozen and only the new head
    trainable. Returns (model, base_model)."""
    base_constructor = BASE_CONSTRUCTORS[architecture]

    # Build architecture without downloading weights
    base_model = base_constructor(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    # Load pretrained ImageNet weights from local .h5 file
    weights_path = config.WEIGHTS_PATHS[architecture]
    if not weights_path.exists():
        raise FileNotFoundError(
            f"ImageNet weights not found at '{weights_path}'. Download the no-top .h5 "
            f"file for {architecture} and place it there, or set the WEIGHTS_DIR "
            f"environment variable to point at your existing copy."
        )
    base_model.load_weights(str(weights_path))
    print(f"Loaded ImageNet weights for {architecture}")

    # Freeze backbone
    base_model.trainable = False

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=f"{architecture}_transfer",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_top_layers(layers_list, fraction: float = 0.30):
    """
    Unfreezes the top `fraction` of `layers_list`, keeping earlier layers
    frozen and BatchNorm layers always frozen (destabilises fine-tuning on
    small medical datasets otherwise).

    Accepts either:
      - base_model.layers (fresh build, phase 1 -> phase 2 transition), or
      - model.layers[:-4] (after reloading a saved model - the flattened
        graph construction in build_transfer_model means the backbone's
        layers are the full model's layers minus the 4 head layers appended
        there: GlobalAveragePooling2D, Dense, Dropout, Dense).
    """
    n_layers = len(layers_list)
    n_frozen = int(n_layers * (1 - fraction))
    for layer in layers_list[:n_frozen]:
        layer.trainable = False
    for layer in layers_list[n_frozen:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    return layers_list


def count_trainable_params(model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


def count_total_params(model) -> int:
    return int(model.count_params())
