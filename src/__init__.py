"""Reusable, independently testable modules for the cancer image classification project.

Submodules:
    data         - dataset loaders, stratified splitting, class-imbalance handling, tf.data pipeline
    augmentation - training-time image augmentation
    models       - transfer-learning model builders
    evaluation   - metrics computation and figure/table reporting
    gradcam      - Grad-CAM heatmap generation and overlay
    training     - two-phase fine-tuning training loop
"""
