import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2

import utils

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

image_path = "./Images/2.jpg"
image = cv2.imread(image_path)

image64 = cv2.resize(image, (64, 64))
image = cv2.cvtColor(image64, cv2.COLOR_BGR2GRAY)
threshold_value = 128
_, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
image = cv2.bitwise_not(image)

image = image.astype("float32") / 255.
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

predicted = encoder.predict(image, batch_size = batch_size)

fig, axes = plt.subplots(2, 13, figsize=(20, 5))
axes[0, 0].imshow(cv2.cvtColor(image64, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Image originale")
axes[0, 0].axis("off")

axes[0, 1].imshow(image[0], cmap="gray")
axes[0, 1].set_title("Image inversée")
axes[0, 1].axis("off")

decoded = decoder.predict(predicted, batch_size = batch_size)

axes[0, 2].imshow(decoded[0], cmap="gray")
axes[0, 2].set_title("Encodé et décodé")
axes[0, 2].axis("off")

axes[1, 0].axis("off")
axes[1, 1].axis("off")
axes[1, 2].axis("off")

classifier = load_model("./Models/Classifieur/classifier.keras")

"""predicted = encoder.predict(decoded, batch_size = batch_size)
predicted = encoder.predict(decoded, batch_size = batch_size)"""
predicted = utils.encoded(decoded, "", encoder, decoder, 2, batch_size, False)

src_class, p, linp = utils.classify(image, classifier)

#utils.pred_bar(linp, fig, axes[0, 1])

axes[0, 1].text(0.5, -0.15, f"({src_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 1].transAxes)

guessed_class, p, linp = utils.classify(decoded, classifier)

axes[0, 2].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 2].transAxes)

mean_encoded_src = encoded_means[src_class]
for dst_class in range(10):
    mean_encoded_dst = encoded_means[dst_class]
    translation = mean_encoded_dst - mean_encoded_src
    translated = predicted + translation
    decoded = decoder.predict(translated, batch_size=batch_size)

    im = axes[0, dst_class + 3].imshow(decoded[0], cmap="gray")
    axes[0, dst_class + 3].set_title(f"Classe {dst_class}")
    axes[0, dst_class + 3].axis("off")

    guessed_class, p, linp = utils.classify(decoded, classifier)

    #utils.pred_bar(linp, fig, axes[0, dst_class + 2])

    axes[0, dst_class + 3].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, dst_class + 3].transAxes)

    reencoded = encoder.predict(decoded, batch_size = batch_size)
    invTranslated = reencoded - translation

    redecoded = decoder.predict(invTranslated, batch_size = batch_size)

    axes[1, dst_class + 3].imshow(redecoded[0], cmap="gray")
    axes[1, dst_class + 3].axis("off")

    guessed_class, p, linp = utils.classify(redecoded, classifier)

    #utils.pred_bar(linp, fig, axes[1, dst_class + 2])

    axes[1, dst_class + 3].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[1, dst_class + 3].transAxes)

#utils.pred_classes(fig)

plt.tight_layout()
plt.savefig("./Results/mnist-translated-image.png")