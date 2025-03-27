import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.mixture import GaussianMixture

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_mean, z_logvar, z_train = latent.encode_n(
    autoencoder,
    x=x_train,
    y=y_train,
    n=2,
    save_cache=True,
    return_dist=True
)

# Uniform

print("Uniforme")

z_min = np.min(z_train, axis=0)
z_max = np.max(z_train, axis=0)

num_samples = 10000
sampled_latents = np.random.uniform(low=z_min, high=z_max, size=(num_samples, z_train.shape[1]))

decoded_images = autoencoder.decoder.predict(sampled_latents)

idx = np.random.choice(len(decoded_images), 100, replace=False)

plt.figure(figsize=(20, 20))
for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(decoded_images[idx[i]].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "RandomSamples" / "mnist-random-samples-uniform.png")

# VAE Distribution

print("Distribution du VAE")

num_samples = 10000
idx = np.random.choice(len(z_mean), num_samples, replace=True)
epsilon = np.random.normal(size=(num_samples, z_mean.shape[1]))
sampled_latents = z_mean[idx] + np.exp(0.5 * z_logvar[idx]) * epsilon
decoded_images = autoencoder.decoder.predict(sampled_latents)

idx = np.random.choice(len(decoded_images), 100, replace=False)

plt.figure(figsize=(20, 20))
for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(decoded_images[idx[i]].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "RandomSamples" / "mnist-random-samples-vaeprior.png")

# Gaussian Mixture

print("GMM")

n_components = 25
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, verbose=True)
gmm.fit(z_train)

num_samples = 10000
sampled_latents, _ = gmm.sample(num_samples)

decoded_images = autoencoder.decoder.predict(sampled_latents)

idx = np.random.choice(len(decoded_images), 100, replace=False)

plt.figure(figsize=(20, 20))
for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(decoded_images[idx[i]].squeeze(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "RandomSamples" / "mnist-random-samples-gm.png")