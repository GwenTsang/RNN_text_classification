"""
train.py

Boucle d'entraînement pour la classification de textes avec un RNN.

Le modèle (architecture + compilation) est défini dans le notebook.
Ici on gère :
- les seeds
- le chargement du corpus
- la création des tf.data.Dataset
- l'adaptation du TextVectorization
- les callbacks
- la boucle model.fit / evaluate
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import load_data, create_dataset, compute_class_weights

# Valeurs par défaut (tu peux les override dans le notebook)
DEFAULT_DATA_DIR = "./Corpus_classif"
DEFAULT_OUTDIR = "./runs/text_cls"
DEFAULT_VAL_SPLIT = 0.20
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 4
DEFAULT_SEED = 42


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_text_classification_model(
    model: tf.keras.Model,
    vectorizer: tf.keras.layers.TextVectorization,
    max_seq_len: int,
    data_dir: str = DEFAULT_DATA_DIR,
    outdir: str = DEFAULT_OUTDIR,
    val_split: float = DEFAULT_VAL_SPLIT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[tf.keras.callbacks.History, float]:
    """Entraîne et évalue un modèle de classification de textes.

    Parameters
    ----------
    model : tf.keras.Model
        Modèle Keras déjà compilé.
        On suppose que la première couche est un TextVectorization
        compatible avec `vectorizer`.
    vectorizer : tf.keras.layers.TextVectorization
        La couche de vectorisation utilisée dans le modèle
        (elle sera adaptée sur le texte d'entraînement).
    max_seq_len : int
        Longueur maximale de séquence utilisée pour construire le modèle
        (uniquement pour l'affichage/logs).
    data_dir : str, default DEFAULT_DATA_DIR
        Dossier racine du corpus.
    outdir : str, default DEFAULT_OUTDIR
        Dossier de sortie pour les runs/logs (créé si nécessaire).
    val_split : float, default DEFAULT_VAL_SPLIT
        Proportion du corpus utilisée pour la validation.
    batch_size : int, default DEFAULT_BATCH_SIZE
        Taille de batch.
    epochs : int, default DEFAULT_EPOCHS
        Nombre maximum d'époques.
    patience : int, default DEFAULT_PATIENCE
        Patience pour l'early stopping (en nombre d'époques).
    seed : int, default DEFAULT_SEED
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    history : keras.callbacks.History
        L'historique renvoyé par model.fit.
    val_accuracy : float
        Accuracy de validation finale.
    """
    # Pour isoler les runs
    tf.keras.backend.clear_session()
    set_seeds(seed)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 1. Chargement des données
    print(f"\n=== Loading data (MAX_SEQ_LEN={max_seq_len}) ===")
    files, labels, class_names = load_data(data_dir)
    num_classes = len(class_names)
    print(f"Found {len(files)} files across {num_classes} classes: {class_names}")

    # 2. Split train / val (stratifié)
    X_train, X_val, y_train, y_val = train_test_split(
        files,
        labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # 3. tf.data.Dataset
    train_ds = create_dataset(X_train, y_train, batch_size, shuffle=True, seed=seed)
    val_ds = create_dataset(X_val, y_val, batch_size, shuffle=False, seed=None)

    # 4. Adaptation du TextVectorization sur le texte d'entraînement
    print("Adapting vectorizer...")
    text_ds = train_ds.map(lambda x, y: x)
    vectorizer.adapt(text_ds)

    # 5. Poids de classes
    class_weights = compute_class_weights(y_train, num_classes)

    # 6. Callbacks (early stopping + reduce LR on plateau)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            verbose=1,
        ),
    ]

    # 7. Entraînement
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Évaluation
    print("\nEvaluating...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"MAX_SEQ_LEN={max_seq_len} | Validation Accuracy: {val_acc:.4f}")

    return history, float(val_acc)
