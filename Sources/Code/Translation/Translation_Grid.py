import numpy as np

import tensorflow as tf
from keras.datasets import mnist

import matplotlib.pyplot as plt

from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from Code.Utils import cache, latent, utils

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_train = x_train.reshape(-1, 28, 28, 1)

x_test = x_test.astype("float32") / 255.
x_test = x_test.reshape(-1, 28, 28, 1)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_test = latent.encode_n(autoencoder, x_test, 3, save_cache=True)
z_class_distributions = latent.class_distributions_n(autoencoder, x_train, y_train, 2, save_cache=True)

digits = [
    1333, # 0
    9415, # 1
    3773, # 2
    524, # 3
    1980, # 4
    1874, # 5
    4252, # 6
    6960, # 7
    8466, # 8
    5333  # 9
]

classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifieur" / "classifier.keras")

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

for src_class in range(10):
    src_digit = digits[src_class]

    for dst_class in range(10):
        z = z_test[src_digit:src_digit + 1]

        z_translated = latent.translate(z, src_class, dst_class, z_class_distributions)

        x_decoded = autoencoder.decoder.predict(z_translated, batch_size=32)
        x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy()

        guessed_class, p, linp = utils.classify(x_decoded, classifier)

        ax = axes[src_class, dst_class]
        
        ax.imshow(x_decoded[0].reshape(28, 28), cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()

plt.savefig(cache.RESULTS_FOLDER / "mnist-translation-grid.png")