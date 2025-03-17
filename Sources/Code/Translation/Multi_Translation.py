import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling  # Important
from Code.Utils import cache, latent, utils

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")
classifier  = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifieur" / "classifier.keras")

digits = np.array([
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],  # 0
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217],   # 1
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032],    # 2
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787],    # 3
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937],  # 4
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428],  # 5
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178],    # 6
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749],     # 7
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385],   # 8
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977]    # 9
])

selected_indices = digits.flatten()
x_selected_test = x_test[selected_indices]

z_test = latent.encode_n(autoencoder, x_selected_test, 3, save_cache=False)

z_class_distributions = latent.class_distributions_n(autoencoder, x_train, y_train, 2, save_cache=True)

all_translated = []
translation_indices = []
batch_counter = 0

original_x_sources = []
original_class_indices = []

for src_class in range(10):
    class_indices = digits[src_class]
    original_class_indices.append(class_indices)
    
    x_source = x_test[class_indices]
    original_x_sources.append(x_source)
    
    start, end = src_class * 10, (src_class + 1) * 10
    z_src = z_test[start:end]
    
    z_src_repeated = np.repeat(z_src, 10, axis=0)
    y_src = np.full((100,), src_class, dtype=int)
    y_dst = np.tile(np.arange(10), 10)
    
    z_translated = latent.translate(z_src_repeated, y_src, y_dst, z_class_distributions)
    all_translated.append(z_translated)
    
    translation_indices.append((batch_counter, batch_counter + z_translated.shape[0]))
    batch_counter += z_translated.shape[0]

all_translated = np.concatenate(all_translated, axis=0)
x_decoded_all = autoencoder.decoder.predict(all_translated)
x_decoded_all = tf.image.resize(x_decoded_all, (28, 28)).numpy()

guessed_all, _, certainties_all = utils.classify(x_decoded_all, classifier)

for src_class in range(10):
    start_idx, end_idx = translation_indices[src_class]
    x_decoded = x_decoded_all[start_idx:end_idx]
    guessed_classes = guessed_all[start_idx:end_idx]
    certainties = certainties_all[start_idx:end_idx]
    
    x_decoded_grid = x_decoded.reshape(10, 10, 28, 28)
    guessed_grid = guessed_classes.reshape(10, 10)
    certainties_grid = certainties.reshape(10, 10)
    
    x_source = original_x_sources[src_class]
    guessed_source, _, certainties_source = utils.classify(x_source, classifier)
    
    fig, axes = plt.subplots(10, 12, figsize=(24, 20))
    fig.subplots_adjust(hspace=0.2)
    
    axes[0, 0].set_title(f"Source ({src_class})", fontsize=26)
    for j in range(10):
        axes[0, j + 2].set_title(str(j), fontsize=26)

    for i, idx in enumerate(original_class_indices[src_class]):
        ax = axes[i, 0]
        src_img = x_test[idx].reshape(28, 28)
        ax.imshow(src_img, cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_source[i]}, {certainties_source[i]:.3f})",
                fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis("off")
    
    for i in range(10):
        axes[i, 1].axis("off")
    
    for i in range(10):
        for dst_class in range(10):
            ax = axes[i, dst_class + 2]
            ax.imshow(x_decoded_grid[i, dst_class].reshape(28, 28), cmap="gray")
            ax.text(0.5, -0.15,
                    f"({guessed_grid[i, dst_class]}, {certainties_grid[i, dst_class]:.3f})",
                    fontsize=14, color="blue", ha="center", transform=ax.transAxes)
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(cache.RESULTS_FOLDER / "TranslationGrids" / f"mnist-translation-grid-{src_class}.png")
    plt.close(fig)