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

X_eval = np.concatenate((X_train, X_valid))
Y_eval = np.concatenate((Y_train, Y_valid))

src_digit = 61256
src_class = Y_eval[src_digit]
dst_class = 9

ae_type = "VAE"

batch_size = 16
latent_dim = 64

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

"""X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}-dis2.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))
X_decoded_all = cache_array(f"{ae_type}-decoded-{latent_dim}-dis2.npy", lambda: decoder.predict(X_encoded_all, batch_size = batch_size))
X_reencoded_all = cache_array(f"{ae_type}-reencoded-{latent_dim}-dis.npy", lambda: encoder.predict(X_decoded_all, batch_size = batch_size))"""

"""X_encoded_all = encoder.predict(X_eval, batch_size = batch_size)
X_decoded_all = decoder.predict(X_encoded_all, batch_size = batch_size)
X_reencoded_all = encoder.predict(X_decoded_all, batch_size = batch_size)"""

digits = [
    61333, # 0
    69415, # 1
    63773, # 2
    60524, # 3
    61980, # 4
    61874, # 5
    64252, # 6
    66960, # 7
    68466, # 8
    65333  # 9
]

"""fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    ax = axes[i]
    ax.imshow(X_eval[digits[i]].reshape(28, 28))
    ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-digits.png")"""

encoded_means = [None] * 10
for i in range(10):
    encoded_means[i] = np.mean(X_reencoded_all[Y_eval == i], axis = 0)
    encoded_means[i] = np.expand_dims(encoded_means[i], axis = 0)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

for src_class in range(10):
    src_digit = digits[src_class]

    for dst_class in range(10):
        X_encoded = X_reencoded_all[src_digit:src_digit + 1]

        mean_encoded_src = encoded_means[src_class]
        mean_encoded_dst = encoded_means[dst_class]

        translation = mean_encoded_dst - mean_encoded_src
        translated = X_encoded + translation

        decoded = decoder.predict(translated, batch_size = batch_size)

        ax = axes[src_class, dst_class]
        decoded = tf.image.resize(decoded, (28, 28))
        ax.imshow(decoded.numpy()[0].reshape(28, 28))
        ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-grid.png")