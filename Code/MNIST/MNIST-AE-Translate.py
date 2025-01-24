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

src_digit = 61256
src_class = Y_eval[src_digit]
dst_class = 9

ae_type = "VAE"

batch_size = 16
latent_dim = 8

encoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-encoder.keras")
decoder = load_model(f"./Models/{ae_type}/mnist-{latent_dim}-decoder.keras")

X_encoded_all = cache_array(f"{ae_type}-encoded-{latent_dim}.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))
X_encoded = X_encoded_all[src_digit:src_digit + 1]
X_encoded_class_src = X_encoded_all[Y_eval == src_class]
X_encoded_class_dst = X_encoded_all[Y_eval == dst_class]

mean_encoded_src = np.mean(X_encoded_class_src, axis=0)
mean_encoded_dst = np.mean(X_encoded_class_dst, axis=0)

translation = mean_encoded_dst - mean_encoded_src
translated = X_encoded + translation

decoded = decoder.predict(translated, batch_size = batch_size)

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(X_eval[src_digit].reshape(28, 28))
plt.axis("off")
plt.title("Source")

plt.subplot(1, 2, 2)
plt.imshow(decoded[0].reshape(28, 28))
plt.axis("off")
plt.title("Décodé")

plt.tight_layout()
plt.savefig("./Results/mnist-translation-decoded.png")

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)
X_eval_encoded_tSNE = cache_array(f"{ae_type}-encoded-tsne-{latent_dim}.npy", lambda: tsne.fit_transform(X_encoded_all))

tsne_src_digit = X_eval_encoded_tSNE[src_digit]
tsne_class1_centroid = X_eval_encoded_tSNE[Y_eval == src_class].mean(axis=0)
tsne_class2_centroid = X_eval_encoded_tSNE[Y_eval == dst_class].mean(axis=0)

tsne_translated_digit = tsne_src_digit + (tsne_class2_centroid - tsne_class1_centroid)

plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_eval_encoded_tSNE[:, 0],
    X_eval_encoded_tSNE[:, 1],
    c=Y_eval,
    cmap="Paired",
    alpha=0.35,
    s=6
)

unique_classes = np.unique(Y_eval)
norm = Normalize(vmin = min(unique_classes), vmax = max(unique_classes))

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

plt.scatter(tsne_class1_centroid[0], tsne_class1_centroid[1], marker="x", color="red", s=100, label="Centroïde source")
plt.scatter(tsne_class2_centroid[0], tsne_class2_centroid[1], marker="x", color="blue", s=100, label="Centroïde destination")

plt.scatter(tsne_src_digit[0], tsne_src_digit[1], marker="+", color="green", s=150, label="Chiffre source")

plt.arrow(tsne_class1_centroid[0], tsne_class1_centroid[1],
          tsne_class2_centroid[0] - tsne_class1_centroid[0],
          tsne_class2_centroid[1] - tsne_class1_centroid[1],
          color="black", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (centres)")

plt.arrow(tsne_src_digit[0], tsne_src_digit[1],
          tsne_translated_digit[0] - tsne_src_digit[0],
          tsne_translated_digit[1] - tsne_src_digit[1],
          color="purple", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (chiffre)")

plt.title("Latent t-SNE")
plt.legend()
plt.tight_layout()
plt.savefig("./Results/mnist-translation-tsne.png")