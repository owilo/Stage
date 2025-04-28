import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import argparse

from code.utils import cache, latent, utils, models, plots

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

parser = argparse.ArgumentParser(description="Matrices de confusion de transformation")
parser.add_argument("-a", action='store_true', help="Inclusion de la perturbation")
parser.add_argument("-t", type=int, default=0, help="Méthode (0 : translation, 1 : translation + normalisation, 2 : transformation)")
args = parser.parse_args()

use_alpha = args.a
transform_method = args.t
if transform_method not in list(range(3)):
    raise ValueError("Méthode de transformation invalide. Choisissez 0, 1 ou 2.")

autoencoder, autoencoder_definition = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "dataset_range": (0, 1)}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))

z_test = latent.encode(
    autoencoder,
    x_test,
    y_test,
    2,
    save_cache=True
)

if autoencoder.decoder.requires_labels(): # CVAE
    z_class_distributions = None
else: # BetaVAE
    z_train = latent.encode(
        autoencoder,
        x=x_train,
        y=y_train,
        n_times=2,
        save_cache=True
    )

    z_class_distributions = latent.class_distributions(z_train, y_train)

def process_class_translations(source_class, z_test, y_test, distributions, result_dir):
    class_mask = (y_test == source_class)
    z_class = z_test[class_mask]
    
    z_src = np.repeat(z_class, 10, axis=0)
    y_src = np.repeat([source_class], len(z_class) * 10)
    y_dst = np.tile(np.arange(10), len(z_class))
    
    """if autoencoder.decoder.requires_labels(): # CVAE
        z_dst = latent.style_class_transform(z_src, y_dst)
    else: # Beta-VAE
        z_dst = latent.translate(z_src, y_src, y_dst, distributions)"""

    alpha = None
    if autoencoder_definition["labels"]:
        z_dst = latent.style_class_transform(z_src, y_dst)
    else:
        if transform_method == 0 or transform_method == 1:            
            z_std = np.array([z_class_distributions[c][1] for c in sorted(z_class_distributions)])

            if use_alpha:
                per_sample_std = z_std[y_src]
                alpha = np.random.normal(0.0, per_sample_std)
            else:
                alpha = np.zeros_like(z_src)

            z_dst = latent.translate(z_src + alpha, y_src, y_dst, z_class_distributions, use_std=transform_method == 1)
        else:            
            z_dst = latent.transform_mg(z_src, y_src, y_dst, z_train, y_train, alpha=alpha)

    x_decoded = autoencoder.decoder.predict(z_dst, batch_size=128)
    x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy()
    
    guessed_labels, _, certainties = utils.classify(x_decoded, classifier)
    forward_cm = confusion_matrix(y_dst, guessed_labels, labels=np.arange(10))
    
    plots.compute_confusion_matrix(
        forward_cm,
        certainties,
        np.arange(10),
        result_dir / f"mnist-translation-confusion-{source_class}.png",
        f"Translation de {source_class} → j"
    )
    
    return forward_cm, x_decoded, y_dst, certainties, alpha

def process_inverse_translations(x_decoded_first, y_dst_labels, source_class, distributions, alpha):
    _, _, z_invdst = autoencoder.encoder.predict(x_decoded_first, batch_size=128)

    y_src = [source_class]*len(z_invdst)

    """if autoencoder.decoder.requires_labels(): # CVAE
        z_dst = latent.style_class_transform(z_reencoded, y_src)
    else: # Beta-VAE
        z_dst = latent.translate(z_reencoded, y_dst_labels, y_src, distributions)"""
    
    if autoencoder_definition["labels"]:
        z_dst = latent.style_class_transform(z_invdst, y_src)
    else:
        if transform_method == 0 or transform_method == 1:            
            z_dst = latent.translate(z_invdst - alpha if alpha is not None else z_invdst, y_dst, y_src, z_class_distributions, use_std=transform_method == 1)
        else:            
            z_dst = latent.transform_mg(z_invdst, y_dst, y_src, z_train, y_train, alpha=-alpha)
        
    x_decoded_final = autoencoder.decoder.predict(z_dst, batch_size=128)
    x_decoded_final = tf.image.resize(x_decoded_final, (28, 28)).numpy()
    guessed_labels, _, certainties = utils.classify(x_decoded_final, classifier)
    
    cm = confusion_matrix([source_class] * len(guessed_labels), guessed_labels, labels=np.arange(10))
    return cm, certainties

FOLDER = cache.RESULTS_FOLDER / "TranslationConfusion"
FOLDER.mkdir(parents=True, exist_ok=True)

combined_cm = np.zeros((10, 10), dtype=int)
inverse_cm = np.zeros((10, 10), dtype=int)

all_forward_certainties = np.array([])
all_inverse_certainties = np.array([])
for src_class in range(10):
    print(f"\nClasse {src_class} :")
    
    forward_cm, x_decoded, y_dst, forward_certainties, alpha = process_class_translations(
        src_class, z_test, y_test, z_class_distributions, FOLDER
    )
    combined_cm += forward_cm
    all_forward_certainties = np.concatenate((all_forward_certainties, forward_certainties))

    inverse_cm_class, inverse_certainties = process_inverse_translations(
        x_decoded, y_dst, src_class, z_class_distributions, alpha
    )
    inverse_cm += inverse_cm_class
    all_inverse_certainties = np.concatenate((all_inverse_certainties, inverse_certainties))

plots.compute_confusion_matrix(
    combined_cm,
    all_forward_certainties,
    np.arange(10),
    FOLDER / "mnist-translation-confusion.png",
    "Translation des classes i → j" # Overall
)

plots.compute_confusion_matrix(
    inverse_cm,
    all_inverse_certainties,
    np.arange(10),
    FOLDER / "mnist-inverse-translation-confusion.png",
    "Translation inverse des classes i → j"
)