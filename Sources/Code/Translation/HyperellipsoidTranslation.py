import tensorflow as tf
from tensorflow import keras
import numpy as np

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

import matplotlib.pyplot as plt

def compute_mean_cov(points):
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    return mean, cov

def inside_hyperellipsoid(point, mean, inv_cov, threshold=1.5):
    diff = np.squeeze(point) - mean
    mahalanobis_distance_sq = diff.T @ inv_cov @ diff
    return mahalanobis_distance_sq <= threshold**2

def percentage_inside(z_test, means, inv_covs, threshold=1.5):
    count_inside = 0
    for point in z_test:
        for mean, inv_cov in zip(means, inv_covs):
            if inside_hyperellipsoid(point, mean, inv_cov, threshold):
                count_inside += 1
                break
    return count_inside / len(z_test) * 100

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_train = latent.encode_n(
    autoencoder,
    x=x_train,
    y=y_train,
    n=2,
    save_cache=True
)

digit = 2102
z_test = latent.encode_n(
    autoencoder,
    x=x_test,
    y=y_test,
    n=3,
    save_cache=True
)

z_class_distributions = latent.class_distributions(z_train, y_train)

means = np.array([v[0] for v in z_class_distributions.values()])
distances = np.linalg.norm(means[:, np.newaxis] - means, axis=2)
length = np.sum(distances) / (means.shape[0] * (means.shape[0] - 1))
length += np.random.rand()

translation = np.random.rand(z_test.shape[-1])
translation = length * np.linalg.norm(translation)

means = []
inv_covs = []

threshold = 14

for src_class in range(10):
    mean, cov = compute_mean_cov(z_train[y_train == src_class])
    inv_cov = np.linalg.inv(cov)
    if not inside_hyperellipsoid(z_test[digit], mean, inv_cov, threshold):
        means.append(mean)
        inv_covs.append(inv_cov)
    else:
        print(f"Le chiffre est dans l'hyperellipsoïde des {src_class}")

percent = percentage_inside(z_test, means, inv_covs, threshold)
print(f"{percent:.2f}% des points sont dans un hyperellipsoïde")

hyperellipsoid_count = len(means)

min_bounds = np.min(z_train, axis=0)
max_bounds = np.max(z_train, axis=0)

max_iter = 1000000

def translate(z_src, translation, max_iter=1000):
    z_dst = np.copy(z_src)
    for i in range(max_iter):
        z_dst = min_bounds + (z_dst + translation - min_bounds) % (max_bounds - min_bounds)

        for j in range(hyperellipsoid_count):
            if inside_hyperellipsoid(z_dst, means[j], inv_covs[j], threshold):
                return True, i, z_dst

    return False, None, None

found_translation, iter_dst, z_dst = translate(z_test[digit: digit + 1], translation, max_iter)
if not found_translation:
    print("La translation n'a pas pu aboutir")
else:
    x_dst = autoencoder.decoder.predict(z_dst)
    
    _, _, z_invdst = autoencoder.encoder.predict(x_dst)
    
    found_inv, iter_inv, z_invsrc = translate(z_invdst, -translation, max_iter)
    if not found_inv:
        print("La translation inverse n'a pas pu aboutir")
    else:
        x_invsrc = autoencoder.decoder.predict(z_invsrc)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(x_test[digit].squeeze(), cmap="gray")
        axs[0].set_title("Source")
        axs[0].axis("off")
        
        axs[1].imshow(x_dst.squeeze(), cmap="gray")
        axs[1].set_title(f"Translation\n({iter_dst}/{max_iter} it.)")
        axs[1].axis("off")
        
        axs[2].imshow(x_invsrc.squeeze(), cmap="gray")
        axs[2].set_title(f"Translation inverse\n({iter_inv}/{max_iter} it.)")
        axs[2].axis("off")
        
        plt.tight_layout()
        plt.show()