import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow as tf

from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_train = latent.encode_n(
    autoencoder,
    x=x_train,
    y=y_train,
    n=2,
    save_cache=True
)

for i in range(10):
    z_class = z_train[y_train == i]

    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='gray')

    plt.boxplot(z_class, patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red', linewidth=3),
            flierprops=dict(markerfacecolor='orange', marker='o', markersize=8, alpha=0.25, markeredgewidth=0))
    
    plt.xlabel("Dimension")
    plt.ylabel("Valeur")
    plt.title(f"Centroïde de la classe {i}")
    plt.tight_layout()
    plt.savefig(cache.RESULTS_FOLDER / "Centroid-Boxplots" / f"mnist-centroid-boxplot-{i}.png")