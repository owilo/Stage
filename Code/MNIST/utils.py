import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches

def cache_array(filename, array_generator, save_cache=True, load_cache=True, verbose=True):
    os.makedirs("./Cache", exist_ok=True)
    file_path = os.path.join("./Cache", filename)
    if not load_cache:
        return array_generator()

    if os.path.exists(file_path):
        if verbose:
            print(f"Chargement des données depuis {filename}")
        return np.load(file_path)
    else:
        if verbose:
            print(f"Fichier {filename} introuvable, génération des données...")
        array = array_generator()
        if (save_cache):
            if verbose:
                print(f"Sauvegarde des données dans {filename}")
            np.save(file_path, array)
        return array
    
def encoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-encoded-{n}.npy", lambda: encoder.predict(
        (decoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_encoding or save_last, name != "", verbose)

def decoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-decoded-{n}.npy", lambda: decoder.predict(
        (encoded(x, name, encoder, decoder, n - 1, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_decoding or save_last, name != "", verbose)

def encoded_means(x, y, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    def calculate_means():
        x_encoded = encoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose)
        encoded_means = [None] * 10
        for i in range(10):
            encoded_means[i] = np.mean(x_encoded[y == i], axis=0)
            encoded_means[i] = np.expand_dims(encoded_means[i], axis=0)
        return np.array(encoded_means)

    return cache_array(f"{name}-{encoder.name}-{decoder.name}-{n}.npy", calculate_means, save_last, verbose)

def classify(image, classifier):
    image = tf.image.resize(image, (28, 28)).numpy()
    pred = classifier.predict(image)
    guessed_class = np.argmax(pred)

    pred_lin = np.zeros_like(pred)
    mask = pred > 0
    pred_lin[mask] = np.log(pred[mask])
    pred_lin -= pred_lin.min()
    pred_lin /= pred_lin.sum()
    return guessed_class, pred, pred_lin

def pred_bar(pred, fig, ax):
    cmap = plt.cm.Paired
    cpred = np.insert(pred.cumsum(), 0, 0)
    norm = BoundaryNorm(cpred, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, aspect=10, orientation="horizontal", pad=0.02, boundaries=cpred, spacing='proportional')
    cbar.ax.xaxis.set_ticks([])

def pred_classes(fig):
    cmap = plt.cm.Paired
    colors = [cmap(i) for i in range(10)]

    legend_patches = [mpatches.Patch(color=colors[i], label=f"{i}") for i in range(10)]

    plt.subplots_adjust(bottom=0.2)
    fig.legend(handles=legend_patches, loc="lower center", ncol=10, fontsize=10, frameon=False, title="Classes")