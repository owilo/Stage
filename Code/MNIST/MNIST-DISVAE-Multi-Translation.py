import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

import utils

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

encoder = load_model("./Models/DISVAE/mnist-256-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-256-decoder.keras")

X_reencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

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

classifier = load_model("./Models/Classifieur/classifier-linp.keras")

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
        Y_pred_proba -= Y_pred_proba.min()
        Y_pred_proba /= Y_pred_proba.sum()

        certainty = np.max(Y_pred_proba)

        ax.imshow(src_image.reshape(28, 28), cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

    for i in range(10):
        axes[i, 1].axis("off")

    for i, src_digit in enumerate(digits[src_class]):
        for dst_class in range(10):
            X_encoded = X_reencoded_valid[src_digit:src_digit + 1]

            mean_encoded_dst = encoded_means[dst_class]

            translation = mean_encoded_dst - mean_encoded_src
            translated = X_encoded + translation

            decoded = decoder.predict(translated, batch_size = batch_size)
            decoded = tf.image.resize(decoded, (28, 28)).numpy()

            Y_pred_proba = classifier.predict(decoded, verbose = False)

            guessed_class = np.argmax(Y_pred_proba)
            Y_pred_proba -= Y_pred_proba.min()
            Y_pred_proba /= Y_pred_proba.sum()

            certainty = np.max(Y_pred_proba)

            ax = axes[i, dst_class + 2]
            ax.imshow(decoded[0].reshape(28, 28), cmap="gray")
            ax.text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./Results/TranslationGrids/mnist-translation-grid-{src_class}.png")