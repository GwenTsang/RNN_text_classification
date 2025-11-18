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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Puisque les classes sont déséquilibrées, on propose, dans un premier temps, de définir, pour chaque classe $k$ un poids $w_k$ donné par :

    \[
    w_k = \frac{N}{K , n_k}
    \]

    Où
    - $N$ est le nombre total de paires $(X_i, y_i)$ dans le jeu de données,
    - $K$ est le nombre total de classes,
    - et $n_k$ est le nombre d’exemples appartenant à la classe $k$ dans le jeu de données (le nombre de fichier txt dans le dossier, ici).
    """)
    return


@app.cell
def _():
    from pathlib import Path

    DATA_DIR = "./corpus_classif"
    root = Path(DATA_DIR)
    return DATA_DIR, Path, root


@app.cell
def _(root):
    import numpy as np

    # classes = noms des sous-dossiers
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
    nom_classe = []

    for cls_name in classes:
        cls_dir = root / cls_name
        for f in cls_dir.iterdir():
            if f.is_file():
                nom_classe.append(class_to_index[cls_name])

    nom_classe = np.array(nom_classe)

    counts = np.bincount(nom_classe)         # nombre d'exemples par classe
    N = len(nom_classe)                      # nombre total d'exemples
    K = len(counts)                      # il y a 7 classes en l'occurence K = 7

    class_weights = {k: N / (K * c) for k, c in enumerate(counts)}

    class_weights_by_name = {
        cls_name: class_weights[class_to_index[cls_name]]
        for cls_name in classes
    }
    classes, counts, class_weights_by_name
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cette fonction est définie de la même manière dans `load_data.py` sous le nom de `compute_class_weights` . Dans le code python qui suit, on l'importe. On importe aussi deux autres manières de pallier le déséquilibre des données (`oversample_data` et `downsample_data`) que l'on va tester juste après.
    """)
    return


@app.cell
def _():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    from load_data import (
        load_data,
        create_dataset,
        compute_class_weights,
        oversample_data,
    )
    from visualization import plot_accuracy

    VOCAB_SIZE   = 30_000
    EMBED_DIM    = 128
    MAX_SEQ_LEN  = 500
    LEARNING_RATE = 1e-3
    BATCH_SIZE   = 512
    VAL_SPLIT    = 0.2
    EPOCHS       = 20
    PATIENCE     = 4
    SEED         = 42
    return (
        BATCH_SIZE,
        EMBED_DIM,
        EPOCHS,
        LEARNING_RATE,
        MAX_SEQ_LEN,
        PATIENCE,
        SEED,
        VAL_SPLIT,
        VOCAB_SIZE,
        compute_class_weights,
        create_dataset,
        load_data,
        oversample_data,
        plot_accuracy,
        tf,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bidirectionnal LSTM
    """)
    return


@app.cell
def _(EMBED_DIM, LEARNING_RATE, MAX_SEQ_LEN, PATIENCE, VOCAB_SIZE, tf):
    def build_bilstm_model(num_classes: int):
        """ Bidirectionnal LSTM text classification model."""
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=MAX_SEQ_LEN,
        )

        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
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


    def make_callbacks():
        return [
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
    return build_bilstm_model, make_callbacks


@app.cell
def train_once(
    BATCH_SIZE,
    EPOCHS,
    SEED,
    VAL_SPLIT,
    compute_class_weights,
    create_dataset,
    make_callbacks,
    plot_accuracy,
    train_test_split,
):
    def train_once(
        files,
        labels,
        num_classes,
        build_model_fn,
        sampler=None,
        use_class_weights=False,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            files,
            labels,
            test_size=VAL_SPLIT,
            stratify=labels,
            random_state=SEED,
        )

        if sampler is not None:
            X_train, y_train = sampler(X_train, y_train, seed=SEED)

        train_ds = create_dataset(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        )
        val_ds = create_dataset(
            X_val, y_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=None,
        )

        # Nous laissons vide "build_model_fn" car on l'utilisera pour définir différents modèles
        model, vectorizer = build_model_fn(num_classes)
        vectorizer.adapt(train_ds.map(lambda x, y: x))

        class_weights = compute_class_weights(y_train) if use_class_weights else None

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(),
        )

        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"Validation accuracy: {val_acc:.4f}")
        plot_accuracy(history)

        return model, history
    return (train_once,)


@app.cell
def _(DATA_DIR, load_data):
    files, labels, class_names = load_data(DATA_DIR)
    num_classes = len(class_names)
    return class_names, files, labels, num_classes


@app.cell
def _(build_bilstm_model, files, labels, num_classes, train_once):
    # Entraînement en utilisant compute_class_weights

    model_lstm, hist_lstm = train_once(
        files, labels, num_classes,
        build_model_fn=build_bilstm_model,
        sampler=None,
        use_class_weights=True,
    )
    return (model_lstm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Maintenant, à la place de `compute_class_weights` nous allons utiliser `oversample_data` qui est une technique de sur-échantillonnage.

    Ici, le dossier qui contient le plus d'éléments est "VisualWork".
    La fonction `oversample_data` fera en sorte que les 6 autres classes du jeu de données aient autant d'éléments. Pour cela, elle tire aléatoirement des exemples déjà existants et les rajoute.

    Cette méthode augmente un peu la charge de calcul et le temps de l'entraînement. On peut en avoir une idée par les calculs qui suivent :
    """)
    return


@app.cell
def _(
    build_bilstm_model,
    files,
    labels,
    num_classes,
    oversample_data,
    train_once,
):
    # Entrainement avec sur-échantillonnage (oversampling)

    model_lstm_2, hist_lstm_2 = train_once(
        files, labels, num_classes,
        build_model_fn=build_bilstm_model,
        sampler=oversample_data,
        use_class_weights=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Résultats :__

    Val accuracy avec class_weights : **0.9157**
    <br>
    Val accyracy avec oversampling : **0.9214**

    L'oversampling semble très légèrement mieux même si le gain d'accuracy est ici inférieur à 0.01.

    Il serait intéressant de répéter ces deux entraînement une dizaine de fois avec des seeds différents pour voir si on constate un écart du même ordre.
    """)
    return


@app.cell
def _(Path, class_names, mo, root):
    max_count = max((len([f for f in d.iterdir() if f.is_file()])
                     for d in Path(root).iterdir() if d.is_dir()), default=0)

    def count_files():
        total = sum(1 for _ in Path(root).rglob("*.txt"))
        return mo.md(f"""
    Dans **l'entrainement sans oversampling** 80% du total des fichiers .txt sont utilisés pendant une seule epoch , soit {int(total*0.8)} fichiers.

    Dans **l'entrainement avec oversampling**, on augmente artificiellement la quantité de données d'entrainement pour atteindre au total {int(max_count*0.8)*len(class_names)} fichiers txt utilisés pendant une seule epoch.
    """)

    count_files()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Configurable_oversampling

    Pour résoudre ce problème, nous avons ajouté à `load_data.py` une fonction `configurable_oversampling` qui permet de paramétrer le nombre d'éléments qu'on veut dans toutes les classes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Influence de la taille du corpus de données d'entrainement pour l'accuracy du Bidirectionnal LSTM
    """)
    return


@app.cell
def _():
    # 5 configurations : on diminue progressivement l(name, train_fraction, val_fraction)
    CONFIGS = [
        ("85/15", 0.85, 0.15),
        ("80/20", 0.80, 0.20),
        ("75/20", 0.75, 0.20),  # on supprime 5% of the dataset
        ("70/20", 0.70, 0.20),  # on supprime 10%
        ("65/20", 0.65, 0.20),  # on supprime 15%
    ]
    SEEDS = [0, 1, 2]  # on répète à chaque fois l'entraînement avec trois seeds différentes
    return CONFIGS, SEEDS


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    build_bilstm_model,
    create_dataset,
    make_callbacks,
    np,
    oversample_data,
    train_test_split,
):
    def train_one_config(files, labels, num_classes, train_frac, val_frac, seed):
        """
        Train the BiLSTM once for a given (train_frac, val_frac, seed),
        """
        if train_frac + val_frac > 1.0:
            raise ValueError("erreur.")

        # Case 1: train + val = 1 (e.g., 85/15, 80/20)
        if np.isclose(train_frac + val_frac, 1.0):
            X_train, X_val, y_train, y_val = train_test_split(
                files,
                labels,
                test_size=val_frac,
                stratify=labels,
                random_state=seed,
            )
        else:
            # Case 2: train + val < 1 -> keep val_frac fixed, downsample train part
            X_tmp, X_val, y_tmp, y_val = train_test_split(
                files,
                labels,
                test_size=val_frac,
                stratify=labels,
                random_state=seed,
            )
            keep_ratio = train_frac / (1.0 - val_frac)  # fraction of X_tmp to keep
            X_train, _, y_train, _ = train_test_split(
                X_tmp,
                y_tmp,
                train_size=keep_ratio,
                stratify=y_tmp,
                random_state=seed,
            )

        # Sur-échantillonner les données d'entraînement
        X_train_bal, y_train_bal = oversample_data(X_train, y_train, seed=seed)

        train_ds = create_dataset(
            X_train_bal, y_train_bal,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
        )
        val_ds = create_dataset(
            X_val, y_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=None,
        )

        model, vectorizer = build_bilstm_model(num_classes)
        vectorizer.adapt(train_ds.map(lambda x, y: x))

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=make_callbacks(),
            verbose=1,
        )

        _, val_acc = model.evaluate(val_ds, verbose=0)
        return val_acc
    return (train_one_config,)


@app.cell
def _(CONFIGS, SEEDS, files, labels, np, num_classes, train_one_config):
    import matplotlib.pyplot as plt

    mean_accs, std_accs, train_fracs = [], [], []

    for name, train_frac, val_frac in CONFIGS:
        print(f"\n TRAIN = {train_frac:.2f}, VAL ={val_frac:.2f})")
        accs = []
        for s in SEEDS:
            print(f"\n--- seed {s} ---")
            acc = train_one_config(files, labels, num_classes, train_frac, val_frac, seed=s)
            accs.append(acc)
            print(f"Validation accuracy (seed {s}): {acc:.4f}")

        accs = np.array(accs)
        mean_acc, std_acc = accs.mean(), accs.std()
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        train_fracs.append(train_frac)

        print(f"\n>>> {name}: mean val acc = {mean_acc:.4f} ± {std_acc:.4f}")


    plt.figure(figsize=(6, 4))
    plt.errorbar(
        np.array(train_fracs) * 100,
        mean_accs,
        yerr=std_accs,
        marker="o",
        linestyle="-",
        capsize=5,
    )
    plt.xlabel("Training set size (% of full dataset)")
    plt.ylabel("Mean validation accuracy (oversampled train)")
    plt.title("Accuracy vs. training data size (BiLSTM + oversampling)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ce code est très long sur CPU (1 heure pour être réalisé complètement sur mon cpu). Si jamais vous n'avez pas de GPU sur votre machine et que vous souhaitez aller plus vite : https://colab.research.google.com/drive/1yhwCoPTSqh1LHcbcdp_SavHeKVkMU1hF?usp=sharing
    """)
    return


@app.cell
def _(mo):
    # Image extraire du colab notebook.
    # Tous les hyperparamètres sont identiques hormis pour MAX_SEQ_LEN   = 550

    mo.image("./plot.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bidirectionnal GRU
    """)
    return


@app.cell
def _(EMBED_DIM, LEARNING_RATE, MAX_SEQ_LEN, VOCAB_SIZE, tf):
    def build_bigru_model(num_classes: int):
        """
        Bidirectional GRU text classifier.
        """
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=MAX_SEQ_LEN,
            name="text_vectorizer",
        )

        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBED_DIM,
                mask_zero=True,
                name="embedding",
            ),
            # First BiGRU layer (sequence → sequence)
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    128,
                    return_sequences=True,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                ),
                name="bigru_1",
            ),
            # Second BiGRU layer (sequence → vector)
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    128,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                ),
                name="bigru_2",
            ),
            tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
            tf.keras.layers.Dropout(0.5, name="dropout_1"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model, vectorizer
    return (build_bigru_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(
    build_bigru_model,
    files,
    labels,
    num_classes,
    oversample_data,
    train_once,
):
    model_gru, hist_gru = train_once(
        files, labels, num_classes,
        build_model_fn=build_bigru_model,
        sampler=oversample_data,
        use_class_weights=False,
    )
    return (model_gru,)


@app.cell
def _(model_gru, model_lstm):
    total_params_lstm = model_lstm.count_params()
    print(total_params_lstm)

    total_params_gru = model_gru.count_params()
    print(total_params_gru)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous avons volontairement choisi un réseau Bi-GRU assez gros pour avoir la meilleure valeur d'accuracy possible.
    Mais cela donne un modèle gourmand en ressources et long à entraîner.

    Ce modèle Bi-GRU a plus de paramètres que le précédent modèle LSTM (4,36 millions _vs_ 4,12 millions) et est pourtant légèrement moins performant que ce dernier (à peu près 0.877 _vs_ 0.915).

    Nous avons constaté dans nos tests que la valeur d'accuracy n'augmente pas de façon linéaire en fonction de la taille du modèle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Petit Bidirectional GRU + GlobalMaxPool
    """)
    return


@app.cell
def _(tf):
    from tensorflow import keras

    GRU_VOCAB_SIZE  = 10_000
    GRU_MAX_SEQ_LEN = 200
    GRU_EMBED_DIM   = 128

    def build_gru_maxpool_model(num_classes: int):
        """
        Modèle Bidirectional GRU + GlobalMaxPool .
        """
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=GRU_VOCAB_SIZE,
            standardize="lower_and_strip_punctuation",
            output_mode="int",
            output_sequence_length=GRU_MAX_SEQ_LEN,
            name="text_vectorizer",
        )

        model = keras.Sequential([
            vectorizer,
            keras.layers.Embedding(
                input_dim=GRU_VOCAB_SIZE,
                output_dim=GRU_EMBED_DIM,
                input_length=GRU_MAX_SEQ_LEN,
                name="embedding",
            ),
            keras.layers.Bidirectional(
                keras.layers.GRU(
                    32,
                    return_sequences=True,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                ),
                name="bigru",
            ),
            keras.layers.GlobalMaxPool1D(name="global_max_pool"),
            keras.layers.Dense(64, activation="relu", name="dense_1"),
            keras.layers.Dropout(0.5, name="dropout_1"),
            keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model, vectorizer
    return (build_gru_maxpool_model,)


@app.cell
def _(build_gru_maxpool_model, files, labels, num_classes, train_once):
    from load_data import configurable_oversampling

    MAX_PER_CLASS = 780

    def oversampler(X, y, seed=None):
        return configurable_oversampling(X, y, max_count=MAX_PER_CLASS, seed=seed)

    model_gru_2, history_gru = train_once(
        files=files,
        labels=labels,
        num_classes=num_classes,
        build_model_fn=build_gru_maxpool_model,  # your BiGRU model
        sampler=oversampler,                      # now capped at 780 / class
        use_class_weights=False,
    )
    print("Total parameters:", model_gru_2.count_params())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il est intéressant de remarquer qu'avec un oversampling qui fait que toutes les classes ont 780 éléments au maximum, les performances d'accuracy semblent à peu près aussi bonnes qu'avec la technique d'oversampling précédente
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LSTM simple 1
    """)
    return


@app.cell
def _(EMBED_DIM, MAX_SEQ_LEN, SEED, VOCAB_SIZE, tf):
    def simple_lstm_model(num_classes: int):
        """
        Modèle LSTM empilé plus léger pour un entraînement sur CPU
        """
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=MAX_SEQ_LEN,
            name="text_vectorizer",
        )

        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBED_DIM,
                mask_zero=True,
                name="embedding",
            ),
            tf.keras.layers.SpatialDropout1D(0.2, name="spatial_dropout"),

            tf.keras.layers.LSTM(128, return_sequences=True, name="lstm_128"),
            tf.keras.layers.LSTM(64, name="lstm_64"),

            tf.keras.layers.BatchNormalization(name="bn_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
            tf.keras.layers.Dropout(0.4, name="dropout_64"),

            tf.keras.layers.BatchNormalization(name="bn_2"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
            tf.keras.layers.Dropout(0.4, name="dropout_32"),

            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-3,
        )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",  # integer labels
            metrics=["accuracy"],
        )

        return model, vectorizer
    return (simple_lstm_model,)


@app.cell
def _(files, labels, num_classes, simple_lstm_model, train_once):
    model_lstm_simple_1, history_lstm_small = train_once(
        files=files,
        labels=labels,
        num_classes=num_classes,
        build_model_fn=simple_lstm_model,
        sampler=None,
        use_class_weights=True,
    )
    return


@app.cell
def _(
    files,
    labels,
    num_classes,
    oversample_data,
    simple_lstm_model,
    train_once,
):
    model_lstm_simple_1, history_lstm_small = train_once(
        files=files,
        labels=labels,
        num_classes=num_classes,
        build_model_fn=simple_lstm_model,
        sampler=oversample_data,
        use_class_weights=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Même si l'entraînement avec oversampling est légèremlent plus long / gourmand, la différence d'accuracy avec un entraînement avec pondération des classes est assez ici assez significative : **0.63** _versus_ **0.83**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LSTM simple 2
    """)
    return


@app.cell
def _(tf):
    def simple_lstm_model_2(num_classes: int):
        VOCAB_SIZE   = 10_000
        EMBED_DIM    = 64
        MAX_SEQ_LEN  = 200
        LEARNING_RATE = 1e-3

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=MAX_SEQ_LEN,
        )

        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3)),
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
    return (simple_lstm_model_2,)


@app.cell
def _(files, labels, num_classes, simple_lstm_model_2, train_once):
    model_lstm_simple_2, history_lstm_small = train_once(
        files=files,
        labels=labels,
        num_classes=num_classes,
        build_model_fn=simple_lstm_model_2,
        sampler=None,
        use_class_weights=True,
    )
    return (model_lstm_simple_2,)


@app.cell
def _(
    files,
    labels,
    num_classes,
    oversample_data,
    simple_lstm_model_2,
    train_once,
):
    model_lstm_simple_2, history_lstm_small_2 = train_once(
        files=files,
        labels=labels,
        num_classes=num_classes,
        build_model_fn=simple_lstm_model_2,
        sampler=oversample_data,
        use_class_weights=False,
    )
    return (model_lstm_simple_2,)


@app.cell
def _(model_lstm_simple_2):
    total_params_simple_lstm_2 = model_lstm_simple_2.count_params()
    print(total_params_simple_lstm_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La différence entre entraînement avec oversampling et entraînement avec class weight est ici moins significative (0.9086 _vs_ 0.8943). Il faudrait, là aussi, répéter l'expérience avec plusieurs seeds pour mesurer la variance.

    Notons que ce modèle LSTM a moins de paramètres que les précédents et obtient de très bonnes performances pour sa taille. Cela se doit notamment aux hyperparamètres que nous avons modifié.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
