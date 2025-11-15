from pathlib import Path

import numpy as np
import tensorflow as tf


def load_data(data_dir="./corpus_classif"):
    """Retourne (files, labels, class_names) à partir de data_dir."""
    data_dir = Path(data_dir)

    files = []
    labels = []
    class_names = []

    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            class_id = len(class_names)
            class_names.append(class_dir.name)
            for file in class_dir.glob("*.txt"):
                files.append(str(file))
                labels.append(class_id)

    return np.array(files), np.array(labels, dtype="int32"), class_names


def create_dataset(files, labels, batch_size=128, shuffle=True, seed=42):
    """tf.data.Dataset à partir de chemins de fichiers et labels."""

    def _load_file(path, label):
        text = tf.io.read_file(path)
        return text, label

    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    if shuffle:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_file, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def compute_class_weights(labels):
    counts = np.bincount(labels)
    total = len(labels)
    num_classes = len(counts)
    return {i: total / (num_classes * c) for i, c in enumerate(counts)}

