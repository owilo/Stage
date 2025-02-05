import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os

from sklearn.manifold import TSNE

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

src_digit = 1303
src_class = Y_valid[src_digit]
dst_class = 3

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

X_encoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 1, batch_size)
X_reencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 2, batch_size)
X_rereencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

mean_encoded_src = encoded_means[src_class]
mean_encoded_dst = encoded_means[dst_class]

translation = mean_encoded_dst - mean_encoded_src

translated1 = X_reencoded_valid[src_digit : src_digit + 1] + translation
translated2 = X_rereencoded_valid[src_digit : src_digit + 1] + translation

translated_decoded1 = decoder.predict(translated1, batch_size = batch_size)
translated_decoded2 = decoder.predict(translated2, batch_size = batch_size)

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(X_valid.numpy()[src_digit].reshape(64, 64))
plt.axis("off")
plt.title("Source")

plt.subplot(1, 3, 2)
plt.imshow(translated_decoded1[0].reshape(64, 64))
plt.axis("off")
plt.title("Décodé")

plt.subplot(1, 3, 3)
plt.imshow(translated_decoded2[0].reshape(64, 64))
plt.axis("off")
plt.title("2x Décodé")

plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-decoded-{src_class}-{dst_class}.png")

fig, axes = plt.subplots(3, 2, figsize=(22, 18))
axes = axes.flatten()

axes[0].plot(mean_encoded_dst[0] - mean_encoded_src[0],
             color="gray", ls="--", lw=0.75,
             label=f"Différence (translation {src_class} → {dst_class})")
axes[0].plot(mean_encoded_src[0],
             color="red", lw=2.25,
             label=f"Centroïde source ({src_class})")
axes[0].plot(mean_encoded_dst[0],
             color="blue", lw=2.25,
             label=f"Centroïde destination ({dst_class})")
axes[0].set_title("Centroïdes")
axes[0].legend(loc="lower left")

axes[1].plot(X_encoded_valid[0] - mean_encoded_src[0],
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde source {src_class})")
axes[1].plot(mean_encoded_src[0],
             color="red", ls="--", lw=1.5,
             label=f"Centroïde source ({src_class})")
axes[1].plot(X_encoded_valid[0],
             color="darkred", lw=2.25,
             label=f"Chiffre source ({src_class})")
axes[1].set_title("Centroïde source et Chiffre source")
axes[1].legend(loc="lower left")

axes[2].plot(translated2[0] - mean_encoded_dst[0],
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde destination {dst_class})")
axes[2].plot(mean_encoded_dst[0],
             color="blue", ls="--", lw=1.5,
             label=f"Centroïde destination ({dst_class})")
axes[2].plot(translated2[0],
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[2].set_title("Centroïde destination et Chiffre décodé & translaté")
axes[2].legend(loc="lower left")

axes[3].plot(translated2[0] - mean_encoded_src[0],
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde source {src_class})")
axes[3].plot(mean_encoded_src[0],
             color="red", ls="--", lw=1.5,
             label=f"Centroïde source ({src_class})")
axes[3].plot(translated2[0],
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[3].set_title("Centroïde source et Chiffre décodé & translaté")
axes[3].legend(loc="lower left")

axes[4].plot(translated2[0] - translated1[0],
             color="gray", ls="--", lw=0.75,
             label="Différence")
axes[4].plot(translated2[0],
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[4].plot(translated1[0],
             color="#32CD32", ls="--", lw=1.5,
             label=f"Chiffre translaté ({dst_class})")
axes[4].set_title("Chiffre translaté et Chiffre décodé & translaté")
axes[4].legend(loc="lower left")

if len(axes) > 5:
    fig.delaxes(axes[5])

for ax in axes[:5]:
    ax.grid(True, which="both")
    ax.axhline(y=0, color="gray")
    ax.set_xlabel("Indice")
    ax.set_ylabel("Valeur")

plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-plot-{src_class}-{dst_class}.png")
