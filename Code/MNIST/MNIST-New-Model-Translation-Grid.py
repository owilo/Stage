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

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_reencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

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

classifier = load_model("./Models/Classifieur/classifier.keras")

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