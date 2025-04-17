import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors

from sklearn.manifold import TSNE

from code.utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder, _ = models.select_model(
    models.list_models(criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)})
)

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

z_test = latent.encode(
    autoencoder,
    x=x_test,
    y=y_test,
    n_times=2,
    save_cache=True
)

y_srcs = np.array([2, 4])

classes = np.unique(y_train)
latent_centroids = np.vstack([
    z_train[y_train == cls].mean(axis=0)
    for cls in classes
])

tsne = TSNE(n_components=2, random_state=1337, max_iter=300)
all_embeddings = tsne.fit_transform(
    np.vstack([z_test, latent_centroids])
)

z_tsne = all_embeddings[: len(z_test)]
centroid_tsne = {
    cls: all_embeddings[len(z_test) + i]
    for i, cls in enumerate(classes)
}

plt.figure(figsize=(8, 8))
plt.scatter(
    z_tsne[:, 0],
    z_tsne[:, 1],
    c=y_test,
    cmap="Paired",
    alpha=0.35,
    s=6
)

norm = Normalize(vmin=classes.min(), vmax=classes.max())
cmap = plt.cm.Paired

for cls in classes:
    plt.scatter([], [], color=cmap(norm(cls)), label=str(cls))

desaturation_factor = 0.5
brightness_factor = 0.75

for y_src in y_srcs:
    src = centroid_tsne[y_src]
    base_rgba = cmap(norm(y_src))
    rgb = base_rgba[:3]
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] *= desaturation_factor
    hsv[2] *= brightness_factor
    desat_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    desat_rgba = (*desat_rgb, base_rgba[3])
    for y_dst, dst in centroid_tsne.items():
        if y_src == y_dst:
            continue
        plt.arrow(
            src[0], src[1],
            dst[0] - src[0],
            dst[1] - src[1],
            color=desat_rgba,
            width=0.05
        )

for cls, centroid in centroid_tsne.items():
    plt.scatter(
        centroid[0], centroid[1],
        color="black",
        marker="x",
        s=100,
        label="Centroid" if cls == classes[0] else None
    )

plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "mnist-translation-vectors.png")