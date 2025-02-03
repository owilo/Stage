import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os

from sklearn.manifold import TSNE

from keras.datasets import mnist

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

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_encoded_train = encoder.predict(X_train, batch_size = batch_size)
X_decoded_train = decoder.predict(X_encoded_train, batch_size = batch_size)
X_reencoded_train = encoder.predict(X_decoded_train, batch_size = batch_size)

X_encoded_valid = encoder.predict(X_valid, batch_size = batch_size)
X_decoded_valid = decoder.predict(X_encoded_valid, batch_size = batch_size)
X_reencoded_valid = encoder.predict(X_decoded_valid, batch_size = batch_size)

digits = [
    1333, # 0
    9415, # 1
    3773, # 2
    524, # 3
    1980, # 4
    1874, # 5
    4252, # 6
    6960, # 7
    8466, # 8
    5333  # 9
]

"""fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    ax = axes[i]
    ax.imshow(X_eval[digits[i]].reshape(28, 28))
    ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-digits.png")"""

classifier = load_model("./Models/Classifieur/classifier.keras")

encoded_means = [None] * 10
for i in range(10):
    encoded_means[i] = np.mean(X_reencoded_train[Y_train == i], axis = 0)
    encoded_means[i] = np.expand_dims(encoded_means[i], axis = 0)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

for src_class in range(10):
    src_digit = digits[src_class]

    for dst_class in range(10):
        X_encoded = X_reencoded_valid[src_digit:src_digit + 1]

        mean_encoded_src = encoded_means[src_class]
        mean_encoded_dst = encoded_means[dst_class]

        translation = mean_encoded_dst - mean_encoded_src
        translated = X_encoded + translation

        decoded = decoder.predict(translated, batch_size = batch_size)
        decoded = tf.image.resize(decoded, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(decoded, verbose = False)

        guessed_class = np.argmax(Y_pred_proba)
        certainty = np.max(Y_pred_proba)

        ax = axes[src_class, dst_class]
        ax.imshow(decoded[0].reshape(28, 28), cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-grid.png")