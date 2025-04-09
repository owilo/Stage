import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from code.utils import cache, latent, utils, models, plots

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

x_train_l, y_train_l, x_train_r, y_train_r = utils.split_dataset(x_train, y_train, 0.5) # Moitié gauche pour le VAE

x_train_r = np.concatenate((x_train_r, x_test))
y_train_r = np.concatenate((y_train_r, y_test))

_, _, x_train_rr, y_train_rr = utils.split_dataset(x_train_r, y_train_r, 0.75) # 75% de gauche pour l'entraînement des classifieurs

x_src, y_src, y_dst = utils.split_src_to_dst(x_train_rr, y_train_rr)

autoencoder, autoencoder_definition = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "dataset_range": (0, 0.5)}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))

trace_classifier, _ = models.select_model(models.list_models(
    criteria={"type": "trace_classifier", "autoencoder": autoencoder_definition["category"], "dataset_range": (0.5, 1)}
))

trace_detector, _ = models.select_model(models.list_models(
    criteria={"type": "trace_detector", "autoencoder": autoencoder_definition["category"], "dataset_range": (0.5, 1)},
))

model_type = "cvae" if autoencoder_definition["labels"] else "betavae"

z_src = latent.encode(
    autoencoder,
    x=x_src,
    y=y_src,
    n_times=3,
    save_cache=True
)

if autoencoder.decoder.requires_labels(): # CVAE
    z_dst = latent.style_class_transform(z_src, y_dst)
else: # Beta-VAE
    # z_class_distributions = latent.encode_class_distributions(
    #     autoencoder,
    #     x=x_train_l,
    #     y=y_train_l,
    #     n_times=2,
    #     save_cache=True
    # )

    # z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions)

    z_train = latent.encode(
        autoencoder,
        x=x_train,
        y=y_train,
        n_times=2,
        save_cache=True
    )

    z_dst = latent.transform_mt(z_src, y_src, y_dst, z_train, y_train)

x_dst = autoencoder.decoder.predict(z_dst)
_, _, z_invdst = autoencoder.encoder.predict(x_dst)

if autoencoder.decoder.requires_labels():
    z_invsrc = latent.style_class_transform(z_invdst, y_src)
else:
    z_invsrc = latent.transform_mt(z_invdst, y_dst, y_src, z_train, y_train)
    # z_invsrc = latent.translate(z_invdst, y_dst, y_src, z_class_distributions)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

y_trans = (y_src != y_dst).astype(int)

FOLDER = cache.RESULTS_FOLDER / "TraceConfusion"
FOLDER.mkdir(parents=True, exist_ok=True)

# Reconnaissance de la classe source sans translation
guessed, _, certainties = utils.classify(x_src, trace_classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"trace-{model_type}-i.png",
    f"Reconnaissance de la classe i (i)"
)

## Translation

# Classification de la translation
guessed, _, certainties = utils.classify(x_dst, classifier)
cm = confusion_matrix(y_dst, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"classif-{model_type}-i-j.png",
    f"Classification de la translation (i → j)"
)

# Reconnaissance de la classe source après translation
guessed, _, certainties = utils.classify(x_dst, trace_classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"trace-{model_type}-i-j.png",
    f"Reconnaissance de la classe i (i → j)"
)

# Détection de la translation
guessed, _, certainties = utils.classify(x_dst, trace_detector)
cm = confusion_matrix(y_trans, guessed, labels=np.arange(2))
plots.compute_confusion_matrix(
    cm,
    certainties,
    ["Non détecté", "Détecté"],
    FOLDER / f"detect-{model_type}-i-j.png",
    f"Détection de la translation (i → j)"    
)

## Translation inverse

# Reconnaissance de la classe de destination après translation inverse
guessed, _, certainties = utils.classify(x_invsrc, trace_classifier)
cm = confusion_matrix(y_dst, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"trace-{model_type}-i-J-i.png",
    f"Reconnaissance de la classe j (i → j → i)"
)

# Reconnaissance de la classe source après translation inverse
guessed, _, certainties = utils.classify(x_invsrc, trace_classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"trace-{model_type}-I-j-i.png",
    f"Reconnaissance de la classe i (i → j → i)"
)

# Classification de la translation inverse
guessed, _, certainties = utils.classify(x_invsrc, classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"classif-{model_type}-i-j-i.png",
    f"Classification (i → j → i)"    
)

# Détection de la translation inverse
guessed, _, certainties = utils.classify(x_invsrc, trace_detector)
cm = confusion_matrix(y_trans, guessed, labels=np.arange(2))
plots.compute_confusion_matrix(
    cm,
    certainties,
    ["Non détecté", "Détecté"],
    FOLDER / f"detect-{model_type}-i-j-i.png",
    f"Détection de la translation (i → j → i)"    
)