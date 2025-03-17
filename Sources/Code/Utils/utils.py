import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches

# todo remove
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

# todo remove
def encoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-encoded-{n}.npy", lambda: encoder.predict(
        (decoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_encoding or save_last, name != "", verbose)

# todo remove
def decoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-decoded-{n}.npy", lambda: decoder.predict(
        (encoded(x, name, encoder, decoder, n - 1, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_decoding or save_last, name != "", verbose)

# todo remove
def encoded_means(x, y, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    def calculate_means():
        x_encoded = encoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose)
        encoded_means = [None] * 10
        for i in range(10):
            encoded_means[i] = np.mean(x_encoded[y == i], axis=0)
            encoded_means[i] = np.expand_dims(encoded_means[i], axis=0)
        return np.array(encoded_means)

    return cache_array(f"{name}-{encoder.name}-{decoder.name}-{n}.npy", calculate_means, save_last, verbose)

# todo remove
def encoded_std(x, y, name, encoder, decoder, n, batch_size=1, save_last=True, save_encoding=False, save_decoding=False, verbose=True):
    def calculate_stds():
        x_encoded = encoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose)
        encoded_stds = [None] * 10
        for i in range(10):
            encoded_stds[i] = np.std(x_encoded[y == i], axis=0, ddof=1)
            encoded_stds[i] = np.expand_dims(encoded_stds[i], axis=0)
        return np.array(encoded_stds)

    return cache_array(f"{name}-{encoder.name}-{decoder.name}-{n}.npy", calculate_stds, save_last, verbose)

"""
def classify(image, classifier):
    if (image.ndim >= 3):
        image = tf.image.resize(image, (28, 28)).numpy()
    pred = classifier.predict(image)
    guessed_class = np.argmax(pred)

    pred_lin = np.zeros_like(pred)
    mask = pred > 0
    pred_lin[mask] = np.log(pred[mask])
    pred_lin -= pred_lin.min()
    pred_lin /= pred_lin.sum()
    return guessed_class, pred, pred_lin"""

# todo modify
def classify_binary(image, classifier):
    if (image.ndim >= 3):
        image = tf.image.resize(image, (28, 28)).numpy()
    pred = classifier.predict(image)
    guessed_class = (pred >= 0.5).astype(int)
    certainty = 1.0 - np.abs(guessed_class - pred)
    return guessed_class, pred, certainty

# todo remove
def pred_bar(pred, fig, ax):
    cmap = plt.cm.Paired
    cpred = np.insert(pred.cumsum(), 0, 0)
    norm = BoundaryNorm(cpred, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, aspect=10, orientation="horizontal", pad=0.02, boundaries=cpred, spacing='proportional')
    cbar.ax.xaxis.set_ticks([])

# todo remove
def pred_classes(fig):
    cmap = plt.cm.Paired
    colors = [cmap(i) for i in range(10)]

    legend_patches = [mpatches.Patch(color=colors[i], label=f"{i}") for i in range(10)]

    plt.subplots_adjust(bottom=0.2)
    fig.legend(handles=legend_patches, loc="lower center", ncol=10, fontsize=10, frameon=False, title="Classes")

def split_dataset(x, y, p, seed=0):
    x = np.array(x)
    y = np.array(y)
    
    rng = np.random.default_rng(seed)
    
    indices_1 = []
    indices_2 = []
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        split_idx = int(len(cls_indices) * p)

        indices_1.extend(cls_indices[:split_idx])
        indices_2.extend(cls_indices[split_idx:])
    
    indices_1 = np.array(indices_1)
    indices_2 = np.array(indices_2)
    
    indices_1 = np.sort(indices_1)
    indices_2 = np.sort(indices_2)
    
    x1, y1 = x[indices_1], y[indices_1]
    x2, y2 = x[indices_2], y[indices_2]
    
    return x1, y1, x2, y2

def group_by_class(x, y):
    return {cls: x[y == cls] for cls in np.unique(y)}

def classify(x, classifier):
    x = np.asarray(x, dtype=np.float32)

    single_image = (x.ndim == 3)
    if single_image:
        x = np.expand_dims(x, axis=0)

    predictions = classifier.predict(x, batch_size=x.shape[0])

    if predictions.shape[-1] == 1:
        guessed = (predictions >= 0.5).astype(int)
        certainty = 1.0 - np.abs(guessed - predictions)

        guessed = np.squeeze(guessed, axis=-1)
        predictions = np.squeeze(predictions, axis=-1)
        certainty = np.squeeze(certainty, axis=-1)
    else:
        guessed = np.argmax(predictions, axis=1)
        certainty = np.max(predictions, axis=1)

    if single_image:
        return guessed[0], predictions[0], certainty[0]
    return guessed, predictions, certainty

def preprocess_dataset(x_train, x_test):
    x_train = x_train.astype("float32") / 255.
    x_train = x_train.reshape(-1, 28, 28, 1)

    x_test = x_test.astype("float32") / 255.
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test