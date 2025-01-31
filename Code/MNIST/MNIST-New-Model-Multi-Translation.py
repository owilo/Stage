import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os

from sklearn.manifold import TSNE

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

def cache_array(filename, array_generator, save_cache=True, verbose=True):
    file_path = os.path.join("./Cache", filename)

    if os.path.exists(file_path):
        if (verbose):
            print(f"Chargement des données depuis {filename}")
        return np.load(file_path)
    else:
        array = array_generator()
        if (save_cache):
            if (verbose):
                print(f"Sauvegarde des données dans {filename}")
            np.save(file_path, array)
        return array

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

"""X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}-dis2.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))
X_decoded_all = cache_array(f"{ae_type}-decoded-{latent_dim}-dis2.npy", lambda: decoder.predict(X_encoded_all, batch_size = batch_size))
X_reencoded_all = cache_array(f"{ae_type}-reencoded-{latent_dim}-dis.npy", lambda: encoder.predict(X_decoded_all, batch_size = batch_size))"""

X_encoded_all = encoder.predict(X_valid, batch_size = batch_size)
X_decoded_all = decoder.predict(X_encoded_all, batch_size = batch_size)
X_reencoded_all = encoder.predict(X_decoded_all, batch_size = batch_size)

digits = [
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],  # 0
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217],  # 1
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032],  # 2
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787],  # 3
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937],  # 4
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428],  # 5
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178],  # 6
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749],  # 7
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385],  # 8
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977]  # 9
]

classifier = load_model("./Models/Classifieur/classifier.keras")

encoded_means = [None] * 10
for i in range(10):
    encoded_means[i] = np.mean(X_reencoded_all[Y_valid == i], axis = 0)
    encoded_means[i] = np.expand_dims(encoded_means[i], axis = 0)

for src_class in range(10):
    fig, axes = plt.subplots(10, 12, figsize=(24, 20))
    fig.subplots_adjust(hspace=0.2)

    axes[0, 0].set_title(f"Source ({src_class})", fontsize=26)
    for i in range(10):
        axes[0, i + 2].set_title(str(i), fontsize=26)

    mean_encoded_src = encoded_means[src_class]

    for i, src_digit in enumerate(digits[src_class]):
        ax = axes[i, 0]
        src_image = X_valid[src_digit:src_digit + 1]
        src_image = tf.image.resize(src_image, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(src_image, verbose = False)

        guessed_class = np.argmax(Y_pred_proba)
        certainty = np.max(Y_pred_proba)

        ax.imshow(src_image.reshape(28, 28), cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

    for i in range(10):
        axes[i, 1].axis("off")

    for i, src_digit in enumerate(digits[src_class]):
        for dst_class in range(10):
            X_encoded = X_reencoded_all[src_digit:src_digit + 1]

            mean_encoded_dst = encoded_means[dst_class]

            translation = mean_encoded_dst - mean_encoded_src
            translated = X_encoded + translation

            decoded = decoder.predict(translated, batch_size = batch_size)
            decoded = tf.image.resize(decoded, (28, 28)).numpy()

            Y_pred_proba = classifier.predict(decoded, verbose = False)

            guessed_class = np.argmax(Y_pred_proba)
            certainty = np.max(Y_pred_proba)

            ax = axes[i, dst_class + 2]
            ax.imshow(decoded[0].reshape(28, 28), cmap="gray")
            ax.text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./Results/TranslationGrids/mnist-translation-grid-{src_class}.png")