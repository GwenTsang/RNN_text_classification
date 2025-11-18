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


def compute_class_weights(labels):
    counts = np.bincount(labels)
    total = len(labels)
    num_classes = len(counts)
    return {i: total / (num_classes * c) for i, c in enumerate(counts)}

def oversample_data(files, labels, seed=42):
    """
    Oversampling (sur-échantillonage) des classes minoritaires pour équilibrer le dataset.

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

def configurable_oversampling(files, labels, max_count=None, seed=42):
    """
    Ré-équilibre les classes par sur-échantillonnage des classes minoritaires
    Si max_count est spécifié, sous-échantillonnage des classes majoritaires.

    - Si max_count est None : on oversample chaque classe jusqu'à la taille
      de la classe majoritaire (comportement initial, sans downsampling).
    - Si max_count est un entier : chaque classe aura EXACTEMENT max_count
      exemples (downsample de la classe si > max_count, oversample si < max_count).

    Retourne (files_balanced, labels_balanced).
    """
    rng = np.random.default_rng(seed)

    files = np.asarray(files)
    labels = np.asarray(labels)

    unique_classes, counts = np.unique(labels, return_counts=True)

    # Cible de taille par classe
    if max_count is None:
        target_count = counts.max()
    else:
        target_count = max_count

    new_files = []
    new_labels = []

    for cls, count in zip(unique_classes, counts):
        idx = np.where(labels == cls)[0]
        n_cls = len(idx)

        # 1) Downsampling si la classe est trop grande
        if n_cls > target_count:
            base_idx = rng.choice(idx, size=target_count, replace=False)
            effective_count = target_count
        else:
            base_idx = idx
            effective_count = n_cls

        cls_files = files[base_idx]
        cls_labels = labels[base_idx]

        # 2) Oversampling si la classe est trop petite
        if effective_count < target_count:
            extra_idx = rng.choice(base_idx, size=target_count - effective_count, replace=True)
            extra_files = files[extra_idx]
            extra_labels = labels[extra_idx]

            cls_files = np.concatenate([cls_files, extra_files])
            cls_labels = np.concatenate([cls_labels, extra_labels])

        new_files.append(cls_files)
        new_labels.append(cls_labels)

    new_files = np.concatenate(new_files)
    new_labels = np.concatenate(new_labels)

    # Mélange final
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
