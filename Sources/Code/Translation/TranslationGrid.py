import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

from Code.Models import BetaVAE, Classifier
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

digit_indices = np.array([
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
])

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "classifier.keras")

z_test = latent.encode_n(
    autoencoder,
    x=x_test[digit_indices],
    y=y_test[digit_indices],
    n=3,
    save_cache=False
)

z_class_distributions = latent.class_distributions_n(
    autoencoder,
    x=x_train,
    y=y_train,
    n=2,
    save_cache=True
)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

y_src = np.repeat(np.arange(10), 10)  # [0, 0, ..., 9, 9]
y_dst = np.tile(np.arange(10), 10)  # [0, 1, ..., 9, 0, 1, ..., 9]

z_src = np.repeat(z_test, 10, axis=0)

z_translated = latent.translate(z_src, y_src, y_dst, z_class_distributions)

x_decoded = autoencoder.decoder.predict(z_translated)
x_decoded = tf.image.resize(x_decoded, (28, 28)).numpy() # todo pour le classifieur

guessed_classes, _, certainties = utils.classify(x_decoded, classifier)

x_decoded = x_decoded.reshape(10, 10, 28, 28)
guessed_classes = guessed_classes.reshape(10, 10)
certainties = certainties.reshape(10, 10)

for src_class in range(10):
    for dst_class in range(10):
        ax = axes[src_class, dst_class]
        ax.imshow(x_decoded[src_class, dst_class], cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_classes[src_class, dst_class]}, {certainties[src_class, dst_class]:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "mnist-translation-grid.png")