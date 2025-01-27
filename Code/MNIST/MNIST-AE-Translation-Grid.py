import numpy as np
import matplotlib.pyplot as plt

import os

from keras.datasets import mnist

import tensorflow.keras.backend as K
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

X_eval = np.concatenate((X_train, X_valid))
Y_eval = np.concatenate((Y_train, Y_valid))

ae_type = "AE"

batch_size = 16
latent_dim = 8

encoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-encoder.keras")
decoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-decoder.keras")

X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))

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

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    ax = axes[i]
    ax.imshow(X_eval[digits[i]].reshape(28, 28))
    ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-digits.png")

encoded_means = [None] * 10
for i in range(10):
    encoded_means[i] = np.mean(X_encoded_all[Y_eval == i], axis = 0)
    encoded_means[i] = np.expand_dims(encoded_means[i], axis = 0)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

for src_class in range(10):
    src_digit = digits[src_class]

    for dst_class in range(10):
        X_encoded = X_encoded_all[src_digit:src_digit + 1]

        mean_encoded_src = encoded_means[src_class]
        mean_encoded_dst = encoded_means[dst_class]

        #mean_encoded_src = mean_encoded_src / np.linalg.norm(mean_encoded_src, axis=0, keepdims=True)
        #mean_encoded_dst = mean_encoded_dst / np.linalg.norm(mean_encoded_dst, axis=0, keepdims=True)

        translation = mean_encoded_dst - mean_encoded_src
        translated = X_encoded + translation

        decoded = decoder.predict(translated, batch_size = batch_size)

        ax = axes[src_class, dst_class]
        ax.imshow(decoded[0].reshape(28, 28))
        ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist-translation-grid.png")