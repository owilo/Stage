import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import argparse

from code.utils import cache, latent, utils, models, plots

parser = argparse.ArgumentParser(description="Matrices de confusion des traces dans la transformation")
parser.add_argument("--name", type=str, help="Nom du fichier")
parser.add_argument("--autoencoder", type=str, default=None, help="Nom de l'autoencodeur utilisé")
parser.add_argument("--classifier", type=str, default=None, help="Nom du classifieur utilisé")
parser.add_argument("--tdetector", type=str, default=None, help="Nom du détecteur de traces utilisé")
parser.add_argument("--tclassifier", type=str, default=None, help="Nom du classifieur de traces utilisé")
parser.add_argument("-a", action='store_true', help="Inclusion de la perturbation")
parser.add_argument("-t", type=int, default=0, help="Méthode (0 : translation, 1 : translation + normalisation, 2 : transformation)")
parser.add_argument("--encode", action='store_true', help="Encoder et décoder les images non transformées")
args = parser.parse_args()

default_autoencoder = args.autoencoder
default_classifier = args.classifier
default_trace_detector = args.tdetector
default_trace_classifier = args.tclassifier
encode = args.encode
use_alpha = args.a
transform_method = args.t
if transform_method not in list(range(3)):
    raise ValueError("Méthode de transformation invalide. Choisissez 0, 1 ou 2.")

autoencoder, autoencoder_definition = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "dataset_range": (0, 0.5)}
), auto_choice=default_autoencoder)

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
), auto_choice=default_classifier)

trace_classifier, _ = models.select_model(models.list_models(
    criteria={"type": "trace_classifier", "autoencoder": autoencoder_definition["category"], "dataset_range": (0.5, 1)}
), auto_choice=default_trace_classifier)

trace_detector, _ = models.select_model(models.list_models(
    criteria={"type": "trace_detector", "autoencoder": autoencoder_definition["category"], "dataset_range": (0.5, 1)},
), auto_choice=default_trace_detector)

model_type = args.name if args.name else ("cvae" if autoencoder_definition["labels"] else "betavae")

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

x_train_l, y_train_l, x_train_r, y_train_r = utils.split_dataset(x_train, y_train, 0.5) # (30k VAE - 30k classifieurs + test)

x_train_r = np.concatenate((x_train_r, x_test))
y_train_r = np.concatenate((y_train_r, y_test))

_, _, x_train_rr, y_train_rr = utils.split_dataset(x_train_r, y_train_r, 0.75) # (30k classifieurs - 10k test)

input_shape = tuple(autoencoder_definition["input_shape"])

x_train_rr = utils.resize(x_train_rr, input_shape)

x_ori, y_ori, x_src, y_src = utils.split_dataset(x_train_rr, y_train_rr, 0.0, seed=None) # (1k inchangés - 9k obscurcis)

if encode:
    z_ori = latent.encode(
        autoencoder,
        x=x_ori,
        y=y_ori,
    )
    x_ori = latent.decode(
        autoencoder,
        z=z_ori,
        y=y_ori,
        num_classes=10,
    )

x_src, y_src, y_dst = utils.split_src_to_dst(x_src, y_src)

z_src = latent.encode(
    autoencoder,
    x=x_src,
    y=y_src,
    n_times=2,
    save_cache=True
)

if autoencoder_definition["labels"]:
    z_dst = latent.style_class_transform(z_src, y_dst)
else:
    if transform_method == 0 or transform_method == 1:
        z_class_distributions = latent.encode_class_distributions(
            autoencoder,
            x=x_train_l,
            y=y_train_l,
            n_times=2,
            save_cache=True
        )
        
        z_std = np.array([z_class_distributions[c][1] for c in sorted(z_class_distributions)])

        if use_alpha:
            per_sample_std = z_std[y_src]
            alpha = np.random.normal(0.0, per_sample_std)
        else:
            alpha = np.zeros_like(z_src)

        z_dst = latent.translate(z_src + alpha, y_src, y_dst, z_class_distributions, use_std=transform_method == 1)
    else:
        z_train_l = latent.encode(
            autoencoder,
            x=x_train_l,
            y=y_train_l,
            n_times=2,
            save_cache=True
        ) # TODO make use of the whole dataset?
        
        alpha = np.random.normal(np.zeros_like(z_src), 0.5) if use_alpha else None

        z_dst = latent.transform_mg(z_src, y_src, y_dst, z_train_l, y_train_l, alpha=alpha)

x_dst = autoencoder.decoder.predict(z_dst)
_, _, z_invdst = autoencoder.encoder.predict(x_dst)

if autoencoder_definition["labels"]:
    z_invsrc = latent.style_class_transform(z_invdst, y_src)
else:
    if transform_method == 0 or transform_method == 1:
        z_invsrc = latent.translate(z_invdst - alpha, y_dst, y_src, z_class_distributions, use_std=transform_method == 1)
    else:
        z_invsrc = latent.transform_mg(z_invdst, y_dst, y_src, z_train_l, y_train_l, alpha=-alpha if alpha is not None else None)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

x_det = np.concatenate((x_ori, x_dst))
x_invdet = np.concatenate((x_ori, x_invsrc))
y_det = np.concatenate((np.zeros_like(y_ori, dtype=bool), np.ones_like(y_dst, dtype=bool)))

FOLDER = cache.RESULTS_FOLDER / "TraceConfusion"
FOLDER.mkdir(parents=True, exist_ok=True)

# Classification sans translation

guessed, _, certainties = utils.classify(x_src, classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"classif-{model_type}-i.png",
    f"Classification (i)"
)

if transform_method == 2:
    guessed, _, certainties = latent.classify_mg(z_src, z_train_l, y_train_l)
    cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
    plots.compute_confusion_matrix(
        cm,
        certainties,
        np.arange(10),
        FOLDER / f"classif-{model_type}-i-mg.png",
        f"Classification QDA (i)"
    )

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
    f"Classification (i → j)"
)

if transform_method == 2:
    guessed, _, certainties = latent.classify_mg(z_dst, z_train_l, y_train_l)
    cm = confusion_matrix(y_dst, guessed, labels=np.arange(10))
    plots.compute_confusion_matrix(
        cm,
        certainties,
        np.arange(10),
        FOLDER / f"classif-{model_type}-i-j-mg.png",
        f"Classification QDA (i → j)"
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
guessed, _, certainties = utils.classify(x_det, trace_detector)
cm = confusion_matrix(y_det, guessed, labels=np.arange(2))
plots.compute_confusion_matrix(
    cm,
    certainties,
    ["Non détecté", "Détecté"],
    FOLDER / f"detect-{model_type}-i-j.png",
    f"Détection de la translation (i → j)"    
)

# Détection de traces avec le classifieur standard
guessed, _, certainties = utils.classify(x_dst, classifier)
cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    FOLDER / f"classif-trace-{model_type}-i-j.png",
    f"Classification (i → j)"
)

if transform_method == 2:
    guessed, _, certainties = latent.classify_mg(z_dst, z_train_l, y_train_l)
    cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
    plots.compute_confusion_matrix(
        cm,
        certainties,
        np.arange(10),
        FOLDER / f"classif-trace-{model_type}-i-j-mg.png",
        f"Classification QDA (i → j)"
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

if transform_method == 2:
    guessed, _, certainties = latent.classify_mg(z_invsrc, z_train_l, y_train_l)
    cm = confusion_matrix(y_src, guessed, labels=np.arange(10))
    plots.compute_confusion_matrix(
        cm,
        certainties,
        np.arange(10),
        FOLDER / f"classif-{model_type}-i-j-i-mg.png",
        f"Classification QDA (i → j → i)"
    )

# Détection de la translation inverse
guessed, _, certainties = utils.classify(x_det, trace_detector)
cm = confusion_matrix(y_det, guessed, labels=np.arange(2))
plots.compute_confusion_matrix(
    cm,
    certainties,
    ["Non détecté", "Détecté"],
    FOLDER / f"detect-{model_type}-i-j-i.png",
    f"Détection de la translation (i → j → i)"    
)