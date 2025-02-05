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

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

X_reencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

classifier = load_model("./Models/Classifieur/classifier.keras")

total_conf_matrix = np.zeros((10, 10), dtype=int)

total_certainties = []
for src_class in range(10):
    certainties = []

    plt.figure(figsize=(10, 8))
    conf_matrix = np.zeros((10, 10), dtype=int)

    digits = X_reencoded_valid[Y_valid == src_class]

    mean_encoded_src = encoded_means[src_class]
    for dst_class in range(10):
        mean_encoded_dst = encoded_means[dst_class]
        translation = mean_encoded_dst - mean_encoded_src
        translated = digits + translation

        decoded = decoder.predict(translated, batch_size=batch_size)
        decoded = tf.image.resize(decoded, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(decoded)
        guessed_classes = np.argmax(Y_pred_proba, axis=1)

        certainty = np.max(Y_pred_proba, axis=1)
        certainties.extend(certainty.tolist())
        total_certainties.extend(certainty.tolist())

        for guessed_class in guessed_classes:
            conf_matrix[dst_class, guessed_class] += 1

    total_conf_matrix += conf_matrix

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    percentages = np.where(row_sums == 0, 0, conf_matrix / row_sums * 100)
    
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    average_certainty = np.mean(certainties)

    plt.suptitle(f"Classe source {src_class}", fontsize=22)
    plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)

    annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])
    sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe cible")

    plt.tight_layout()
    plt.savefig(f"./Results/TranslationConfusion/mnist-translation-confusion-{src_class}.png")

accuracy = np.trace(total_conf_matrix) / np.sum(total_conf_matrix)
average_certainty = np.mean(total_certainties)

plt.figure(figsize=(10, 8))
plt.suptitle("Toutes classes source", fontsize=22)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)

row_sums_total = total_conf_matrix.sum(axis=1, keepdims=True)
percentages_total = np.where(row_sums_total == 0, 0, total_conf_matrix / row_sums_total * 100)
annot_total = np.array([["{:.2f}%".format(val) for val in row] for row in percentages_total])
sns.heatmap(percentages_total, annot=annot_total, fmt="", cmap="BuPu")
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")

plt.tight_layout()
plt.savefig(f"./Results/TranslationConfusion/mnist-translation-confusion-all.png")
