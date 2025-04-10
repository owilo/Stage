import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist

from code.utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

latent_dim = z_train.shape[1]

num_classes = 10
num_cols = 10

fig, axes = plt.subplots(num_classes, num_cols, figsize=(num_cols, num_classes))
for i in range(num_classes):
    z_mean = np.mean(z_train[y_train == i], axis=0)
    z_std = np.std(z_train[y_train == i], axis=0)

    for j in range(num_cols):
        z_sample = z_mean + z_std * np.random.randn(1, latent_dim)

        generated_img = autoencoder.decoder(z_sample).numpy()

        axes[i, j].imshow(generated_img.squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "RandomSamples" / "normal-class-samples.png")

fig, axes = plt.subplots(num_classes, num_cols, figsize=(num_cols, num_classes))
for i in range(num_classes):
    z_mean = np.mean(z_train[y_train == i], axis=0)
    z_cov = np.cov(z_train[y_train == i].T)

    for j in range(num_cols):
        z_sample = np.random.multivariate_normal(z_mean, z_cov, size=1)

        generated_img = autoencoder.decoder(z_sample).numpy()

        axes[i, j].imshow(generated_img.squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "RandomSamples" / "mg-class-samples.png")