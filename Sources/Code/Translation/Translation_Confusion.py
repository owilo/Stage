import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from Code.Utils import cache, latent, utils

np.random.seed(42)

# Load data and models
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifieur" / "classifier.keras")

# Encode entire test set once
z_test = latent.encode_n(autoencoder, x_test, 3, save_cache=False)
z_class_distributions = latent.class_distributions_n(autoencoder, x_train, y_train, 2, save_cache=True)

def compute_and_save_confusion(cm, classes, filename, title):
    """Save a precomputed confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def process_class_translations(source_class, z_test, y_test, distributions, result_dir):
    """Process translations and return both results and decoded images."""
    class_mask = (y_test == source_class)
    z_class = z_test[class_mask]
    y_class = y_test[class_mask]
    
    # Prepare translations
    z_translations = np.repeat(z_class, 10, axis=0)
    y_src = np.repeat([source_class], len(z_class) * 10)
    y_dst = np.tile(np.arange(10), len(z_class))
    
    # Translate and decode
    z_translated = latent.translate(z_translations, y_src, y_dst, distributions)
    x_decoded = autoencoder.decoder.predict(z_translated, batch_size=128)
    x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy()
    
    # Forward classification
    guessed_labels, _, _ = utils.classify(x_decoded, classifier)
    forward_cm = confusion_matrix(y_dst, guessed_labels, labels=np.arange(10))
    
    # Save per-class confusion matrix
    compute_and_save_confusion(
        forward_cm,
        np.arange(10),
        result_dir / f"mnist-translation-confusion-{source_class}.png",
        f"Translation from {source_class} (True=Target)"
    )
    
    return forward_cm, x_decoded, y_dst

def process_inverse_translations(x_decoded_first, y_dst_labels, source_class, distributions):
    _, _, z_reencoded = autoencoder.encoder.predict(x_decoded_first, batch_size=128)
    
    z_inverse_trans = latent.translate(z_reencoded, y_dst_labels, [source_class]*len(z_reencoded), distributions)
    
    x_decoded_final = autoencoder.decoder.predict(z_inverse_trans, batch_size=128)
    x_decoded_final = tf.image.resize(x_decoded_final, (28, 28)).numpy()
    guessed_labels, _, _ = utils.classify(x_decoded_final, classifier)
    
    return confusion_matrix([source_class]*len(guessed_labels), guessed_labels, labels=np.arange(10))

forward_dir = cache.RESULTS_FOLDER / "TranslationConfusion"
forward_dir.mkdir(parents=True, exist_ok=True)

combined_cm = np.zeros((10, 10), dtype=int)
inverse_cm = np.zeros((10, 10), dtype=int)

for src_class in range(10):
    print(f"\nClasse {src_class} :")
    
    forward_cm, x_decoded, y_dst = process_class_translations(
        src_class, z_test, y_test, z_class_distributions, forward_dir
    )
    combined_cm += forward_cm
    
    # Process inverse translations using existing decoded images
    inverse_cm_class = process_inverse_translations(
        x_decoded, y_dst, src_class, z_class_distributions
    )
    inverse_cm += inverse_cm_class

# Save final matrices
compute_and_save_confusion(
    combined_cm,
    np.arange(10),
    forward_dir / "mnist-translation-confusion.png",
    "Translation des classes i → j"
)

compute_and_save_confusion(
    inverse_cm,
    np.arange(10),
    forward_dir / "mnist-inverse-translation-confusion.png",
    "Translation inverse des classes i → j"
)