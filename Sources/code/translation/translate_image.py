import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.manifold import TSNE

import cv2

from code.models import *
from code.utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "CVAE" / "cvae128.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "classifier.keras")

"""model_type = "cvae" if autoencoder.decoder.requires_labels() else "betavae"

trace_classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / f"trace-classifier-{model_type}.keras")
trace_detector = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / f"trace-detector-{model_type}.keras")"""

image_path = cache.IMAGES_FOLDER / "2.jpg"
image = cv2.imread(image_path)

image64 = cv2.resize(image, (28, 28))
image = cv2.cvtColor(image64, cv2.COLOR_BGR2GRAY)
threshold_value = 128
_, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
image = cv2.bitwise_not(image)

image = image.astype("float32") / 255.
image = np.expand_dims(image, axis=-1)
x_ori = np.expand_dims(image, axis=0)

y_ori, _, crt_ori = utils.classify(x_ori, classifier)

_, _, z_ori = autoencoder.encoder.predict(x_ori)

x_src = latent.decode(autoencoder, z_ori, tf.keras.utils.to_categorical(y_ori, num_classes=10))

guessed_src, _, crt_src = utils.classify(x_src, classifier)

_, _, z_src = autoencoder.encoder.predict(x_src)

z_src = np.repeat(z_src, 10, axis=0)
y_src = np.full(10, y_ori)
y_dst = np.arange(10)

if autoencoder.decoder.requires_labels(): # CVAE
    z_dst = latent.style_class_transform(z_src, y_dst, num_classes=10)
else: # Beta-VAE
    z_class_distributions = latent.encode_class_distributions(
        autoencoder,
        x=x_train,
        y=y_train,
        n_times=2,
        save_cache=True
    )

    z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions)

x_dst = autoencoder.decoder.predict(z_dst)

guessed_dst, _, crt_dst = utils.classify(x_dst, classifier)

_, _, z_invdst = autoencoder.encoder.predict(x_dst)

if autoencoder.decoder.requires_labels(): # CVAE
    z_invsrc = latent.style_class_transform(z_invdst, y_src, num_classes=10)
else: # Beta-VAE
    z_invsrc = latent.translate(z_invdst, y_dst, y_src, z_class_distributions)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

guessed_invsrc, _, crt_invsrc = utils.classify(x_invsrc, classifier)

fig, axes = plt.subplots(2, 13, figsize=(20, 5))
axes[0, 0].imshow(cv2.cvtColor(image64, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Image originale")
axes[0, 0].axis("off")

axes[0, 1].imshow(x_ori[0], cmap="gray")
axes[0, 1].set_title("Image seuillée")
axes[0, 1].axis("off")
axes[0, 1].text(0.5, -0.15, f"({y_src[0]}, {crt_ori[0]:.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 1].transAxes)

axes[0, 2].imshow(x_src[0], cmap="gray")
axes[0, 2].set_title("Encodé et décodé")
axes[0, 2].axis("off")
axes[0, 2].text(0.5, -0.15, f"({guessed_src[0]}, {crt_src[0]:.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 2].transAxes)

axes[1, 0].axis("off")
axes[1, 1].axis("off")
axes[1, 2].axis("off")

for dst_class in range(10):
    axes[0, dst_class + 3].imshow(x_dst[dst_class], cmap="gray")
    axes[0, dst_class + 3].set_title(f"Classe {dst_class}")
    axes[0, dst_class + 3].axis("off")
    axes[0, dst_class + 3].text(0.5, -0.15, f"({guessed_dst[dst_class]}, {crt_dst[dst_class]:.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, dst_class + 3].transAxes)

    axes[1, dst_class + 3].imshow(x_invsrc[dst_class], cmap="gray")
    axes[1, dst_class + 3].axis("off")
    axes[1, dst_class + 3].text(0.5, -0.15, f"({guessed_invsrc[dst_class]}, {crt_invsrc[dst_class]:.3f})", fontsize=14, color="blue", ha="center", transform=axes[1, dst_class + 3].transAxes)

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "mnist-translated-image.png")