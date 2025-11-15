import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Téléchargement des données depuis le Drive de l'Inalco :
    """)
    return


@app.cell
def _():
    import requests
    import zipfile
    import tarfile
    import io
    import shutil
    import os

    url = "https://drive.inalco.fr/public.php/dav/files/mkkTzCM29BXe4RF/?accept=zip"
    r = requests.get(url, stream=True)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        chemin_dossier = "Partages M2/2 - classification de documents en français - point de départ/"
        fichier_tgz = None

        for file in z.namelist():
            if file.startswith(chemin_dossier) and file.endswith('.tgz'):
                fichier_tgz = file
                break

        if fichier_tgz:
            z.extract(fichier_tgz)

            # Décompresser le .tgz dans le dossier courant
            with tarfile.open(fichier_tgz, 'r:gz') as tar:
                tar.extractall('./')

            # Supprimer le fichier .tgz et les dossiers intermédiaires
            os.remove(fichier_tgz)
            shutil.rmtree('Partages M2')

            print("Téléchargement terminé")
    return


@app.cell
def _():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    from load_data import load_data, create_dataset, compute_class_weights
    from visualization import plot_accuracy

    DATA_DIR = "./corpus_classif"
    VOCAB_SIZE = 30_000
    EMBED_DIM = 128
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    VAL_SPLIT = 0.2
    EPOCHS = 20
    PATIENCE = 4
    SEED = 42


    def build_model(vocab_size, num_classes, max_seq_len):
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_seq_len,
        )

        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, dropout=0.3)
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model, vectorizer


    # --- 1) données
    files, labels, class_names = load_data(DATA_DIR)
    num_classes = len(class_names)
    max_seq_len = 500

    X_train, X_val, y_train, y_val = train_test_split(
        files,
        labels,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=labels,
    )

    train_ds = create_dataset(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    val_ds = create_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False, seed=None)

    # --- 2) modèle
    model, vectorizer = build_model(VOCAB_SIZE, num_classes, max_seq_len)

    # adapter le vectorizer sur le texte d'entraînement
    text_ds = train_ds.map(lambda x, y: x)
    vectorizer.adapt(text_ds)

    # --- 3) class weights + callbacks
    class_weights = compute_class_weights(y_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE // 2,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    plot_accuracy(history)
    return (tf,)


@app.cell
def _(tf):

    print("Physical devices:", tf.config.list_physical_devices())
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les fichiers py qui suivent visent à faire varier l'hyperparamètre `sequence length` et à voir les variations de l'accuracy provoquées
    """)
    return


if __name__ == "__main__":
    app.run()
