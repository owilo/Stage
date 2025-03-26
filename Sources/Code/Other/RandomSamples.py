import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.mixture import GaussianMixture

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

np.random.seed(1337)
tf.keras.utils.set_random_seed(1337)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

_, _, z = autoencoder.encoder.predict(x_train)

n_components = 25
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=1337, verbose=True)
gmm.fit(z)

num_samples = 100
sampled_latents, _ = gmm.sample(num_samples)

decoded_images = autoencoder.decoder.predict(sampled_latents)

plt.figure(figsize=(20, 20))
for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(decoded_images[i].reshape(64, 64))
    plt.gray()
    ax.axis("off")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "mnist-random-samples.png")