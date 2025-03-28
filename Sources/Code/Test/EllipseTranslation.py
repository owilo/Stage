import tensorflow as tf
from tensorflow import keras
import numpy as np
import umap
from scipy.stats import chi2

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

def compute_mean_cov(points):
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    return mean, cov

def inside_ellipse(point, mean, inv_cov, threshold=1.5):
    diff = np.squeeze(point) - mean
    mahalanobis_distance_sq = diff.T @ inv_cov @ diff
    return mahalanobis_distance_sq <= threshold**2

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

digit = 2103
z_test = latent.encode_n(
    autoencoder,
    x=x_test,
    y=y_test,
    n=3,
    save_cache=True
)

ump = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
z_umap_train = ump.fit_transform(z_train)

means = []
inv_covs = []

for src_class in range(10):
    mean, cov = compute_mean_cov(z_umap_train[y_train == src_class])
    inv_cov = np.linalg.inv(cov)
    means.append(mean)
    inv_covs.append(inv_cov)

means = np.array(means)
inv_covs = np.array(inv_covs)

z_class_distributions = latent.class_distributions(z_train, y_train)

class_means = np.array([v[0] for v in z_class_distributions.values()])
distances = np.linalg.norm(class_means[:, np.newaxis] - class_means, axis=2)
length = np.sum(distances) / (class_means.shape[0] * (class_means.shape[0] - 1))
length += np.random.rand()

translation = np.random.rand(z_test.shape[-1])
translation = length * np.linalg.norm(translation)

threshold = np.sqrt(chi2.ppf(0.75, df=2)) # 80% des points devraient être dans l'ellipse de leur classe

z_src = z_test[digit: digit + 1]
x_src = autoencoder.decoder.predict(z_src)

def candidate_ellipses(means, inv_covs, z_umap_src):
    candidate_means = []
    candidate_inv_covs = []
    for i in range(len(means)):
        if not inside_ellipse(z_umap_src, means[i], inv_covs[i], threshold):
            candidate_means.append(means[i])
            candidate_inv_covs.append(inv_covs[i])
    return candidate_means, candidate_inv_covs

min_bounds = np.min(z_train, axis=0)
max_bounds = np.max(z_train, axis=0)

def translate(z_src, translation, means, inv_covs, max_iter=1000):
    z_dst = np.copy(z_src)
    path = [z_dst.copy()]
    for i in range(max_iter):
        z_dst = min_bounds + (z_dst + translation - min_bounds) % (max_bounds - min_bounds)
        path.append(z_dst.copy())
        z_umap_dst = ump.transform(z_dst)
        for j in range(len(means)):
            if inside_ellipse(z_umap_dst, means[j], inv_covs[j], threshold):
                return True, i + 1, z_dst, path
    return False, None, None, path

z_umap_src = ump.transform(z_src)
forward_means, forward_inv_covs = candidate_ellipses(means, inv_covs, z_umap_src)
found_translation, iter_dst, z_dst, latent_path = translate(z_src, translation, forward_means, forward_inv_covs)
if not found_translation:
    print("La translation n'a pas pu aboutir")
else:
    print("Translation trouvée")
    x_dst = autoencoder.decoder.predict(z_dst)
    
    _, _, z_invdst = autoencoder.encoder.predict(x_dst)
    
    z_umap_invdst = ump.transform(z_invdst)
    inverse_means, inverse_inv_covs = candidate_ellipses(means, inv_covs, z_umap_invdst)
    found_inv, iter_inv, z_invsrc, inv_latent_path = translate(z_invdst, -translation, inverse_means, inverse_inv_covs)
    if not found_inv:
        print("La translation inverse n'a pas pu aboutir")
    else:
        print("Translation inverse trouvée")
        x_invsrc = autoencoder.decoder.predict(z_invsrc)
        
        umap_path = np.array([ump.transform(z) for z in latent_path]).squeeze()
        umap_inv_path = np.array([ump.transform(z) for z in inv_latent_path]).squeeze()
        
        cmap = plt.get_cmap('Paired')
        colors = [cmap(i) for i in range(10)]

        fig, ax = plt.subplots(figsize=(8, 8))

        for digit_class in range(10):
            idx = np.where(y_train == digit_class)
            ax.scatter(z_umap_train[idx, 0], z_umap_train[idx, 1],
                    s=10, color=colors[digit_class], alpha=0.1, label=f"{digit_class}")

        ax.plot(umap_path[:, 0], umap_path[:, 1], color='red', lw=2, marker='o', markersize=3, label="Translation")
        ax.plot(umap_inv_path[:, 0], umap_inv_path[:, 1], color='green', lw=2, marker='o', markersize=3, label="Translation Inverse")

        def plot_cov_ellipse(mean, cov, ax, nstd=threshold, **kwargs):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * nstd * np.sqrt(vals)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
            ax.add_patch(ellipse)

        for mean, inv_cov in zip(means, inv_covs):
            cov = np.linalg.inv(inv_cov)
            plot_cov_ellipse(mean, cov, ax, nstd=threshold, edgecolor='blue', fc='None', lw=2)

        ax.set_title("Trajectoire sur l'espace UMAP")
        ax.legend()
        plt.savefig(cache.RESULTS_FOLDER / f"mnist-ellipse-translation-umap-{digit}.png")
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        axs[0].imshow(x_test[digit].squeeze(), cmap="gray")
        axs[0].set_title("Source")
        axs[0].axis("off")
        
        axs[1].imshow(x_src.squeeze(), cmap="gray")
        axs[1].set_title("Image Encodée")
        axs[1].axis("off")
        
        axs[2].imshow(x_dst.squeeze(), cmap="gray")
        axs[2].set_title(f"Translation\n({iter_dst} it.)")
        axs[2].axis("off")
        
        axs[3].imshow(x_invsrc.squeeze(), cmap="gray")
        axs[3].set_title(f"Translation Inverse\n({iter_inv} it.)")
        axs[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(cache.RESULTS_FOLDER / f"mnist-ellipse-translation-{digit}.png")
