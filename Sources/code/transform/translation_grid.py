import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

from code.utils import cache, latent, utils, models

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

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "dataset_range": (0, 1)}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))


z_test = latent.encode(
    autoencoder,
    x=x_test[digit_indices],
    y=y_test[digit_indices],
    n_times=2,
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

y_src = np.repeat(np.arange(10), 10)  # [0, 0, ..., 9, 9]
y_dst = np.tile(np.arange(10), 10)  # [0, 1, ..., 9, 0, 1, ..., 9]

z_src = np.repeat(z_test, 10, axis=0)

if autoencoder.decoder.requires_labels(): # CVAE
    z_dst = latent.style_class_transform(z_src, y_dst)
else: # Beta-VAE
    z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions, use_std=False)

x_dst = autoencoder.decoder.predict(z_dst)
_, _, z_invdst = autoencoder.encoder.predict(x_dst)

if autoencoder.decoder.requires_labels(): # CVAE
    z_invsrc = latent.style_class_transform(z_invdst, y_src)
else: # Beta-VAE
    z_invsrc = latent.translate(z_invdst, y_dst, y_src, z_class_distributions, use_std=False)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

x_dst = utils.resize(x_dst, (28, 28)) # todo pour le classifieur
x_invsrc = utils.resize(x_invsrc, (28, 28))

guessed_classes_dst, _, certainties_dst = utils.classify(x_dst, classifier)
guessed_classes_invsrc, _, certainties_invsrc = utils.classify(x_invsrc, classifier)

x_dst = x_dst.reshape(10, 10, 28, 28)
guessed_classes_dst = guessed_classes_dst.reshape(10, 10)
certainties_dst = certainties_dst.reshape(10, 10)

x_invsrc = x_invsrc.reshape(10, 10, 28, 28)
guessed_classes_invsrc = guessed_classes_invsrc.reshape(10, 10)
certainties_invsrc = certainties_invsrc.reshape(10, 10)

fig, axes = plt.subplots(10, 11, figsize=(22, 20))
for src_class in range(10):
    ax = axes[src_class, 0]
    ax.imshow(x_test[digit_indices][src_class], cmap="gray")
    ax.axis('off')

    for dst_class in range(10):
        ax = axes[src_class, dst_class + 1]
        ax.imshow(x_dst[src_class, dst_class], cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_classes_dst[src_class, dst_class]}, {certainties_dst[src_class, dst_class]:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
fig.canvas.draw()

col0_right = axes[0, 0].get_position().x1
col1_left = axes[0, 1].get_position().x0
line_x = (col0_right + col1_left) / 2

fig.add_artist(plt.Line2D([line_x, line_x], [0, 1], color='red', linewidth=4, transform=fig.transFigure))
plt.savefig(cache.RESULTS_FOLDER / "mnist-translation-grid.png")


fig, axes = plt.subplots(10, 11, figsize=(22, 20))
for src_class in range(10):
    ax = axes[src_class, 0]
    ax.imshow(x_test[digit_indices][src_class], cmap="gray")
    ax.axis('off')

    for dst_class in range(10):
        ax = axes[src_class, dst_class + 1]
        ax.imshow(x_invsrc[src_class, dst_class], cmap="gray")
        ax.text(0.5, -0.15, f"({guessed_classes_invsrc[src_class, dst_class]}, {certainties_invsrc[src_class, dst_class]:.3f})", fontsize=14, color="blue", ha="center", transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
fig.canvas.draw()

col0_right = axes[0, 0].get_position().x1
col1_left = axes[0, 1].get_position().x0
line_x = (col0_right + col1_left) / 2

fig.add_artist(plt.Line2D([line_x, line_x], [0, 1], color='red', linewidth=4, transform=fig.transFigure))
plt.savefig(cache.RESULTS_FOLDER / "mnist-inverse-translation-grid.png")