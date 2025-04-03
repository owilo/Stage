import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

from code.models import CVAE, Classifier
from code.utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "CVAE" / "cvae16.keras")
Classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "classifier.keras")

digits = np.array([
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217],
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032],
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787],
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937],
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428],
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178],
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749],
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385],
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977]
])

original_x_sources = [x_test[class_indices] for class_indices in digits]
x_source_all = np.concatenate(original_x_sources)
guessed_src, _, cert_src = utils.classify(x_source_all, Classifier)
guessed_sources = guessed_src.reshape(10, 10)
certainties_sources = cert_src.reshape(10, 10)

selected_indices = digits.flatten()
z_test = latent.encode(
    autoencoder, 
    x=x_test[selected_indices],
    y=y_test[selected_indices],
    n_times=3,
    save_cache=False
)

if autoencoder.decoder.requires_labels(): # CVAE
    z_class_distributions = None
else: # BetaVAE
    z_class_distributions = latent.encode_class_distributions(
        autoencoder,
        x=x_train,
        y=y_train,
        n_times=2,
        save_cache=True
    )

z_src_all = np.repeat(z_test, 10, axis=0)
y_src_all = np.repeat(np.arange(10), 100)
y_dst_all = np.tile(np.arange(10), 100)

#z_translated = latent.translate(z_src_all, y_src_all, y_dst_all, z_class_distributions)

if autoencoder.decoder.requires_labels(): # CVAE
    z_translated = latent.style_class_transform(z_src_all, y_dst_all)
else: # Beta-VAE
    z_translated = latent.translate(z_src_all, y_src_all, y_dst_all, z_class_distributions)

x_decoded = autoencoder.decoder.predict(z_translated)

_, _, z_reencoded = autoencoder.encoder.predict(x_decoded)

#z_inv_translated = latent.translate(z_reencoded, y_dst_all, y_src_all, z_class_distributions)

if autoencoder.decoder.requires_labels(): # CVAE
    z_inv_translated = latent.style_class_transform(z_reencoded, y_src_all)
else: # Beta-VAE
    z_inv_translated = latent.translate(z_reencoded, y_dst_all, y_src_all, z_class_distributions)

x_reconstructed = autoencoder.decoder.predict(z_inv_translated)

def generate_and_save_grids(x_decoded, filename_suffix):
    x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy()
    guessed, _, certainties = utils.classify(x_decoded, Classifier)
    
    for src_class in range(10):
        class_indices = digits[src_class]
        src_images = x_test[class_indices]
        
        start_idx = src_class * 100
        class_translations = x_decoded[start_idx:start_idx+100]
        class_guessed = guessed[start_idx:start_idx+100]
        class_certainty = certainties[start_idx:start_idx+100]
        
        fig, axes = plt.subplots(10, 12, figsize=(24, 20), 
                               constrained_layout=True, dpi=80)
        axes[0, 0].set_title(f"Source ({src_class})", fontsize=26)
        
        for j in range(10):
            axes[0, j + 2].set_title(str(j), fontsize=26)
        
        for i in range(10):
            axes[i, 1].axis("off")
            ax = axes[i, 0]
            ax.imshow(src_images[i].reshape(28, 28), cmap="gray")
            ax.axis("off")
            ax.text(0.5, -0.15, 
                    f"({guessed_sources[src_class, i]}, {certainties_sources[src_class, i]:.3f})",
                    fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        
        translation_grid = class_translations.reshape(10, 10, 28, 28)
        guessed_grid = class_guessed.reshape(10, 10)
        cert_grid = class_certainty.reshape(10, 10)
        
        for row in range(10):
            for col in range(10):
                ax = axes[row, col + 2]
                ax.imshow(translation_grid[row, col], cmap="gray")
                ax.axis("off")
                ax.text(0.5, -0.15,
                        f"({guessed_grid[row, col]}, {cert_grid[row, col]:.3f})",
                        fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        
        file = cache.RESULTS_FOLDER / "TranslationGrids" / f"mnist-{filename_suffix}-{src_class}.png"
        plt.savefig(file, bbox_inches='tight')
        plt.close(fig)
        print(f"Sauvegarde du fichier {file}")

generate_and_save_grids(x_decoded, "translation-grid")
generate_and_save_grids(x_reconstructed, "inverse-translation-grid")