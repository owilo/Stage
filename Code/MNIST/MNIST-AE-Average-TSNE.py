import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os

from sklearn.manifold import TSNE

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

batch_size = 16

ae_type = "VAE"
latent_dims = [8, 32, 64, 128]
dest_dim = 128

means = [[None] * 10] * len(latent_dims)
for i, latent_dim in enumerate(latent_dims):
    encoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-encoder.keras")
    decoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-decoder.keras")

    X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))
    X_decoded_all = cache_array(f"{ae_type}-decoded-{latent_dim}.npy", lambda: decoder.predict(X_encoded_all, batch_size = batch_size))
    X_reencoded_all = cache_array(f"{ae_type}-reencoded-{latent_dim}.npy", lambda: encoder.predict(X_decoded_all, batch_size = batch_size))

    for j in range(10):
        means[i][j] = decoder.predict(np.expand_dims(np.mean(X_reencoded_all[Y_eval == j], axis = 0), 0))[0]

encoder = load_model(f"./Models/{ae_type}/mnist-{dest_dim}-encoder.keras")
decoder = load_model(f"./Models/{ae_type}/mnist-{dest_dim}-decoder.keras")

X_pretrained = np.copy(X_eval)
for i in range(len(latent_dims)):
    X_pretrained = np.concatenate((X_pretrained, means[i]))

X_encoded_all = cache_array(f"{ae_type}-p-encoded-{dest_dim}.npy", lambda: encoder.predict(X_pretrained, batch_size = batch_size))
X_decoded_all = cache_array(f"{ae_type}-p-decoded-{dest_dim}.npy", lambda: decoder.predict(X_encoded_all, batch_size = batch_size))
X_reencoded_all = cache_array(f"{ae_type}-p-reencoded-{dest_dim}.npy", lambda: encoder.predict(X_decoded_all, batch_size = batch_size))

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)

X_eval_encoded_average_tSNE = cache_array(f"{ae_type}-encoded-translated-average-tsne-{latent_dim}.npy", lambda: tsne.fit_transform(X_reencoded_all))
X_eval_encoded_tSNE = X_eval_encoded_average_tSNE[:len(X_eval)]

plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_eval_encoded_tSNE[:, 0],
    X_eval_encoded_tSNE[:, 1],
    c=Y_eval,
    cmap="Paired",
    alpha=0.35,
    s=6
)

cmap = plt.get_cmap("Paired", len(latent_dims))
colors = ["r", "g", "b", "y"]

for i in reversed(range(len(latent_dims))):
    for j in range(10):
        pos = X_eval_encoded_average_tSNE[len(X_eval) + i * 10 + j]
        dcolor = tuple(c * 0.6 for c in cmap(i)[:3])
        plt.scatter(pos[0], pos[1], marker="x", color=colors[i], s = 150 + i * 25)

plt.title("Latent t-SNE")
#plt.legend()
plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-average-tsne-{dest_dim}.png")