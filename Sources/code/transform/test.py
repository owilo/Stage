import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from code.utils import models

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "category": "SCVAE"}
))

# 1) Load MNIST test set and preprocess
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test[..., None].astype("float32") / 255.0
y_test = keras.utils.to_categorical(y_test, num_classes=10).astype("float32")

# Select one example per class (0–9)
selected_idxs = [np.where(y_test[:, i] == 1)[0][0] for i in range(10)]
images = x_test[selected_idxs]
labels = y_test[selected_idxs]

z_mean, z_logvar, z, w_prob = autoencoder.encoder.predict(images)
labels = np.argmax(w_prob, axis=-1)
w = tf.one_hot(labels, depth=10).numpy()

reconstructed = autoencoder.decoder.predict([z, w])
_, _, _, w_probp = autoencoder.encoder.predict(reconstructed)
labelsp = np.argmax(w_probp, axis=-1)
wp = tf.one_hot(labelsp, depth=10).numpy()

def format_w(vec):
    return int(np.argmax(vec))

# 1) FIG1: Originals vs Reconstructions (10 rows × 2 cols)
fig1, axes1 = plt.subplots(nrows=10, ncols=2, figsize=(4, 20))
for i in range(10):
    for j, img in enumerate([images[i], reconstructed[i]]):
        ax = axes1[i, j]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        # annotate w below each image
        ax.text(0.5, -0.25, format_w(wp[i] if j == 1 else w[i]), fontsize=14, color="blue", ha="center", transform=ax.transAxes)
fig1.tight_layout()
fig1.savefig('fig1.png', dpi=150)
plt.close(fig1)

# 2a) FIG2A: Style swap grid (rows: original class w_i, cols: style z_j)
fig2a, axes2a = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        ax = axes2a[i, j]
        z_j = z[j:j+1]
        w_i = w[i:i+1]
        wp_i = wp[i:i+1]
        img_ij = autoencoder.decoder.predict([z_j, w_i])[0]
        ax.imshow(img_ij.squeeze(), cmap='gray')
        ax.axis('off')
        # annotate w_i under each
        ax.text(0.5, -0.25, format_w(wp_i[0]), fontsize=14, color="blue", ha="center", transform=ax.transAxes)
fig2a.tight_layout()
fig2a.savefig('fig2a.png', dpi=150)
plt.close(fig2a)

# 2b) FIG2B: Same as 2a but soft probabilities (rows: original class w_i, cols: style z_j)
fig2b, axes2b = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        ax = axes2b[i, j]
        z_j = z[j:j+1]
        w_i = w_prob[i:i+1]
        wp_i = wp[i:i+1]
        img_ij = autoencoder.decoder.predict([z_j, w_i])[0]
        ax.imshow(img_ij.squeeze(), cmap='gray')
        ax.axis('off')
        # annotate w_i under each
        ax.text(0.5, -0.25, format_w(wp_i[0]), fontsize=14, color="blue", ha="center", transform=ax.transAxes)
fig2b.tight_layout()
fig2b.savefig('fig2b.png', dpi=150)
plt.close(fig2b)

print("Saved: fig1.png, fig2a.png, fig2b.png with w annotations")