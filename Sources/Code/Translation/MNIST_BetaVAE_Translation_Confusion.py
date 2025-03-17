import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf

from Code.Utils import cache, latent, utils

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifieur" / "classifier.keras")

z_test = latent.encode_n(autoencoder, x_test, 3, save_cache=True)
z_class_distributions = latent.class_distributions_n(autoencoder, x_train, y_train, 2, save_cache=True)

total_conf_matrix = np.zeros((10, 10), dtype=int)

total_certainties = []
for src_class in range(10):
    certainties = []

    conf_matrix = np.zeros((10, 10), dtype=int)

    digits = X_reencoded_valid[Y_valid == src_class]

    #mean_encoded_src = encoded_means[src_class]
    for dst_class in range(10):
        #mean_encoded_dst = encoded_means[dst_class]
        #translation = mean_encoded_dst - mean_encoded_src
        translated = encoded_means[dst_class] + (encoded_std[dst_class] / encoded_std[src_class]) * (digits - encoded_means[src_class])

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

    plt.figure(figsize=(10, 8))
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
