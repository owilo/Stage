import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn.metrics import confusion_matrix

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

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

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_encoded_train = encoder.predict(X_train64, batch_size = batch_size)
X_decoded_train = decoder.predict(X_encoded_train, batch_size = batch_size)
X_reencoded_train = encoder.predict(X_decoded_train, batch_size = batch_size)

X_encoded_valid = encoder.predict(X_valid64, batch_size = batch_size)
X_decoded_valid = decoder.predict(X_encoded_valid, batch_size = batch_size)
X_reencoded_valid = encoder.predict(X_decoded_valid, batch_size = batch_size)

encoded_means = [None] * 10
for i in range(10):
    encoded_means[i] = np.mean(X_reencoded_train[Y_train == i], axis = 0)
    encoded_means[i] = np.expand_dims(encoded_means[i], axis = 0)

classifier = load_model("./Models/Classifieur/classifier.keras")

Y_pred = classifier.predict(X_valid)
guessed_classes = np.argmax(Y_pred, axis=1)

mnist_cm = confusion_matrix(Y_valid, guessed_classes)

total_conf_matrix = np.zeros((10, 10), dtype=int)
total_diff_matrix = np.zeros((10, 10), dtype=int)

total_certainties = []
for dst_class in range(10):
    certainties = []

    plt.figure(figsize=(10, 8))
    conf_matrix = np.zeros((10, 10), dtype=int)

    mean_encoded_dst = encoded_means[dst_class]
    for src_class in range(10):
        mean_encoded_src = encoded_means[src_class]
        
        digits = X_reencoded_valid[src_class == Y_valid]

        translation = mean_encoded_dst - mean_encoded_src
        translated = digits + translation

        decoded = decoder.predict(translated, batch_size = batch_size)

        reencoded = encoder.predict(decoded, batch_size = batch_size)

        invTranslated = reencoded - translation

        redecoded = decoder.predict(invTranslated, batch_size = batch_size)
        redecoded = tf.image.resize(redecoded, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(redecoded)

        guessed_classes = np.argmax(Y_pred_proba, axis=1)

        certainty = np.max(Y_pred_proba, axis=1)
        certainties.extend(certainty.tolist())
        total_certainties.extend(certainty.tolist())

        for guessed_class in guessed_classes:
            conf_matrix[src_class, guessed_class] += 1

    total_conf_matrix += conf_matrix

    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    average_certainty = np.mean(certainties)
    
    plt.suptitle(f"Classe destination {dst_class}", fontsize=22)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="BuPu")
    plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe cible")

    plt.tight_layout()
    plt.savefig(f"./Results/TranslationConfusion/mnist-inverted-translation-confusion-{dst_class}.png")

    diff_matrix = conf_matrix - mnist_cm
    total_diff_matrix += diff_matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_matrix, annot=True, fmt="d", cmap="PiYG", center=0)
    plt.suptitle(f"Différence - Classe destination {dst_class}", fontsize=22)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe cible")
    
    plt.tight_layout()
    plt.savefig(f"./Results/TranslationConfusion/mnist-diff-inverted-translation-confusion-{dst_class}.png")


accuracy = np.trace(total_conf_matrix) / np.sum(total_conf_matrix)
average_certainty = np.mean(total_certainties)

plt.figure(figsize=(10, 8))
plt.suptitle("Toutes classes destination", fontsize=22)
sns.heatmap(total_conf_matrix, annot=True, fmt="d", cmap="BuPu")
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")

plt.tight_layout()
plt.savefig("./Results/TranslationConfusion/mnist-inverted-translation-confusion-all.png")

diff_matrix = conf_matrix - mnist_cm
total_diff_matrix += diff_matrix

plt.figure(figsize=(10, 8))
sns.heatmap(total_diff_matrix, annot=True, fmt="d", cmap="PiYG", center=0)
plt.suptitle("Différence - Toutes classes destination", fontsize=22)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")

plt.tight_layout()
plt.savefig("./Results/TranslationConfusion/mnist-diff-inverted-translation-confusion-all.png")