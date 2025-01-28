import numpy as np
import matplotlib.pyplot as plt

import os

from keras.datasets import mnist

from umap import UMAP

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

def cache_array(filename, array_generator, save_cache=True, verbose=True):
    file_path = os.path.join("./Cache", filename)

    if os.path.exists(file_path):
        if verbose:
            print(f"Chargement des données depuis {filename}")
        return np.load(file_path)
    else:
        array = array_generator()
        if save_cache:
            if verbose:
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
latent_dims = [8, 32, 64, 128]

digit = 1

umap = UMAP(n_components=2, random_state=1337)
plt.figure(figsize=(10, 8))

cmap = plt.cm.get_cmap("Paired", len(latent_dims))
colors = [cmap(i) for i in range(len(latent_dims))]

centroids = [None] * len(latent_dims)
for i, latent_dim in enumerate(latent_dims):
    encoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-encoder.keras")

    X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}.npy", lambda: encoder.predict(X_eval, batch_size=batch_size))
    X_encoded_digit = X_encoded_all[Y_eval == digit]

    X_umap = umap.fit_transform(X_encoded_digit)

    plt.scatter(X_umap[:, 0], X_umap[:, 1], color=colors[i], label=f"{latent_dim}", alpha=0.35, s=6)

    centroids[i] = np.mean(X_umap, axis=0)

for i, latent_dim in enumerate(latent_dims):
    dcolor = tuple(c * 0.6 for c in colors[i][:3])
    plt.scatter(centroids[i][0], centroids[i][1], color=dcolor, marker='x', s=250, linewidths=4, label=f"Centroïde {latent_dim}")

plt.title(f"UMAP {digit}")
plt.legend()
plt.grid(True)
plt.savefig(f"./Results/mnist-umap-overlap-{digit}.png")