import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

latent_dim = z.shape[1]

def build_empirical_cdf_functions(latents):
    cdf_functions = []
    n = latents.shape[0]
    for i in range(latents.shape[1]):
        sorted_vals = np.sort(latents[:, i])
        
        cdf = lambda z, sorted_vals=sorted_vals, n=n: np.searchsorted(sorted_vals, z, side='right') / n
        inv = lambda u, sorted_vals=sorted_vals: np.percentile(sorted_vals, u * 100)
        
        cdf_functions.append((cdf, inv))
    return cdf_functions

cdf_functions = build_empirical_cdf_functions(z)

def map_to_uniform(z, cdf_functions):
    u = np.array([cdf_functions[i][0](z[i]) for i in range(len(z))])
    return u

def map_from_uniform(u, cdf_functions):
    z = np.array([cdf_functions[i][1](u[i]) for i in range(len(u))])
    return z

def translate_in_uniform(u, c):
    return (u + c) % 1.0

sample_img = x_test[0:1]
z_test, _, _ = autoencoder.encoder.predict(sample_img)
z_test = z_test.flatten()

u_test = map_to_uniform(z_test, cdf_functions)

c = 0.5
u_translated = translate_in_uniform(u_test, c)

z_translated = map_from_uniform(u_translated, cdf_functions)

decoded_img = autoencoder.decoder.predict(z_translated.reshape(1, -1))

c = 0.5

fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))

for i in range(10):
    for j in range(10):
        idx = i * 5 + j
        sample_img = x_test[idx:idx+1]  
        z_test, _, _ = autoencoder.encoder.predict(sample_img)
        z_test = z_test.flatten()

        u_test = map_to_uniform(z_test, cdf_functions)
        u_translated = translate_in_uniform(u_test, c)
        z_translated = map_from_uniform(u_translated, cdf_functions)

        decoded_img = autoencoder.decoder.predict(z_translated.reshape(1, -1))

        ax = axes[i * 2, j]
        ax.imshow(sample_img.squeeze(), cmap="gray")
        ax.axis("off")

        ax = axes[i * 2 + 1, j]
        ax.imshow(decoded_img.squeeze().reshape(64, 64), cmap="gray")
        ax.axis("off")

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "mnist-cdf-translation.png")