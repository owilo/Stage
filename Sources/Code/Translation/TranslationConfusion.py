import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Code.Training.CVAE import CVAE, Encoder, Decoder, Sampling # Important
#from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from Code.Training.Classifier import Classifier # Important
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "CVAE" / "cvae16.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "classifier.keras")

z_test = latent.encode_n(autoencoder, x_test, y_test, 3, save_cache=False)

if autoencoder.decoder.requires_labels(): # CVAE
    z_class_distributions = None
else: # BetaVAE
    z_class_distributions = latent.class_distributions_n(
        autoencoder,
        x=x_train,
        y=y_train,
        n=2,
        save_cache=True
    )

def compute_confusion_matrix(cm, certainties, labels, filename, title_prefix=""):
    accuracy = np.trace(cm) / np.sum(cm)
    avg_certainty = np.mean(certainties)

    percentages = (cm / np.sum(cm, axis=1)) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{percentages[i, j]:.1f}%"

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=100.0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.xlabel("Classe prédite", fontsize=12)
    plt.ylabel("Classe cible", fontsize=12)
    plt.suptitle(title_prefix, fontsize=18)
    plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {avg_certainty:.2%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion '{filename}' sauvegardée")

def process_class_translations(source_class, z_test, y_test, distributions, result_dir):
    class_mask = (y_test == source_class)
    z_class = z_test[class_mask]
    
    z_src = np.repeat(z_class, 10, axis=0)
    y_src = np.repeat([source_class], len(z_class) * 10)
    y_dst = np.tile(np.arange(10), len(z_class))
    
    if autoencoder.decoder.requires_labels(): # CVAE
        z_dst = latent.style_class_transform(z_src, y_dst)
    else: # Beta-VAE
        z_dst = latent.translate(z_src, y_src, y_dst, distributions)

    x_decoded = autoencoder.decoder.predict(z_dst, batch_size=128)
    x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy()
    
    guessed_labels, _, certainties = utils.classify(x_decoded, classifier)
    forward_cm = confusion_matrix(y_dst, guessed_labels, labels=np.arange(10))
    
    compute_confusion_matrix(
        forward_cm,
        certainties,
        np.arange(10),
        result_dir / f"mnist-translation-confusion-{source_class}.png",
        f"Translation de {source_class} → j"
    )
    
    return forward_cm, x_decoded, y_dst, certainties

def process_inverse_translations(x_decoded_first, y_dst_labels, source_class, distributions):
    _, _, z_reencoded = autoencoder.encoder.predict(x_decoded_first, batch_size=128)

    y_src = [source_class]*len(z_reencoded)

    if autoencoder.decoder.requires_labels(): # CVAE
        z_dst = latent.style_class_transform(z_reencoded, y_src)
    else: # Beta-VAE
        z_dst = latent.translate(z_reencoded, y_dst_labels, y_src, distributions)
    
    #z_inverse_trans = latent.translate(z_reencoded, y_dst_labels, y_src, distributions)
    
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
    
    forward_cm, x_decoded, y_dst, forward_certainties = process_class_translations(
        src_class, z_test, y_test, z_class_distributions, FOLDER
    )
    combined_cm += forward_cm
    all_forward_certainties = np.concatenate((all_forward_certainties, forward_certainties))

    inverse_cm_class, inverse_certainties = process_inverse_translations(
        x_decoded, y_dst, src_class, z_class_distributions
    )
    inverse_cm += inverse_cm_class
    all_inverse_certainties = np.concatenate((all_inverse_certainties, inverse_certainties))

compute_confusion_matrix(
    combined_cm,
    all_forward_certainties,
    np.arange(10),
    FOLDER / "mnist-translation-confusion.png",
    "Translation des classes i → j"
)

compute_confusion_matrix(
    inverse_cm,
    all_inverse_certainties,
    np.arange(10),
    FOLDER / "mnist-inverse-translation-confusion.png",
    "Translation inverse des classes i → j"
)