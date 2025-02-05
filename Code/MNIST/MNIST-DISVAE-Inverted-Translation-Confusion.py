import numpy as np
import seaborn as sns
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

X_train64 = tf.image.resize(X_train, (64, 64))
X_valid64 = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

X_reencoded_valid = utils.encoded(X_valid64, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train64, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

classifier = load_model("./Models/Classifieur/classifier.keras")

conf_matrix = np.zeros((10, 10), dtype=int)

for src_class in range(10):
    mean_encoded_src = encoded_means[src_class]

    digits = X_reencoded_valid[src_class == Y_valid]

    certainties = []
    for dst_class in range(10):
        mean_encoded_dst = encoded_means[dst_class]

        translation = mean_encoded_dst - mean_encoded_src
        translated = digits + translation

        decoded = decoder.predict(translated, batch_size=batch_size)

        reencoded = encoder.predict(decoded, batch_size=batch_size)

        invTranslated = reencoded - translation

        redecoded = decoder.predict(invTranslated, batch_size=batch_size)
        redecoded = tf.image.resize(redecoded, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(redecoded)

        guessed_classes = np.argmax(Y_pred_proba, axis=1)

        certainties.append(np.max(Y_pred_proba, axis=1))

        for guessed_class in guessed_classes:
            conf_matrix[src_class, guessed_class] += 1

row_sums = conf_matrix.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, conf_matrix / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
average_certainty = np.mean(certainties)

plt.figure(figsize=(10, 8))
plt.suptitle(f"Translation inverse", fontsize=22)
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu")
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")

plt.tight_layout()
plt.savefig(f"./Results/TranslationConfusion/mnist-inverted-translation-confusion.png")