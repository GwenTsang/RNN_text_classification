from pathlib import Path

import numpy as np
import tensorflow as tf


def load_data(data_dir="/content/corpus_classif"):
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


def oversample_data(files, labels, seed=42):
    """
    Oversampling des classes minoritaires pour équilibrer le dataset.

    Retourne (files_oversampled, labels_oversampled)
    """
    rng = np.random.default_rng(seed)

    unique_classes, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    new_files = []
    new_labels = []

    for cls, count in zip(unique_classes, counts):
        idx = np.where(labels == cls)[0]

        # On prend tous les exemples déjà existants
        cls_files = files[idx]
        cls_labels = labels[idx]

        # S'il manque des exemples pour atteindre max_count, on en rajoute par tirage avec remise
        if count < max_count:
            extra_indices = rng.choice(idx, size=max_count - count, replace=True)
            extra_files = files[extra_indices]
            extra_labels = labels[extra_indices]

            cls_files = np.concatenate([cls_files, extra_files])
            cls_labels = np.concatenate([cls_labels, extra_labels])

        new_files.append(cls_files)
        new_labels.append(cls_labels)

    new_files = np.concatenate(new_files)
    new_labels = np.concatenate(new_labels)

    # On mélange l’ensemble oversamplé
    perm = rng.permutation(len(new_files))
    return new_files[perm], new_labels[perm]

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


