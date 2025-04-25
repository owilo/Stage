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

(x_train, y_train), (_, _) = mnist.load_data()
x_train, = utils.preprocess_dataset(x_train)

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

z_class_distributions = latent.class_distributions(z_train, y_train)

z_mean = np.array([z_class_distributions[c][0] for c in sorted(z_class_distributions)])

z_all, configuration = utils.pack_arrays(z_train, z_mean)

tsne = TSNE(n_components=2, random_state=1337, perplexity=75)

z_tsne = tsne.fit_transform(z_all)
y_srcs = np.array([2, 4])

z_train_tsne, z_mean_tsne = utils.unpack_arrays(z_tsne, configuration)

unique_classes = np.unique(y_train)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
cmap = plt.cm.Paired

plt.figure(figsize=(8, 8))
plt.scatter(
    z_train_tsne[:, 0],
    z_train_tsne[:, 1],
    c=y_train,
    cmap="Paired",
    alpha=0.35,
    s=6
)

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

plt.scatter(
    z_mean_tsne[:, 0], z_mean_tsne[:, 1],
    color="black",
    marker="x",
    s=100,
    label="Centroid"
)

plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-centroids-tsne.png")

plt.figure(figsize=(8, 8))
plt.scatter(
    z_train_tsne[:, 0],
    z_train_tsne[:, 1],
    c=y_train,
    cmap="Paired",
    alpha=0.35,
    s=6
)

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

desaturation_factor = 0.55
brightness_factor = 0.75

for y_src in y_srcs:
    src = z_mean_tsne[y_src]
    base_rgba = cmap(norm(y_src))
    rgb = base_rgba[:3]
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] *= desaturation_factor
    hsv[2] *= brightness_factor
    desat_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    desat_rgba = (*desat_rgb, base_rgba[3])
    for y_dst, dst in enumerate(z_mean_tsne):
        if y_src == y_dst:
            continue
        plt.arrow(
            src[0], src[1],
            dst[0] - src[0],
            dst[1] - src[1],
            color=desat_rgba,
            width=0.5,
            length_includes_head=True
        )

plt.scatter(
    z_mean_tsne[:, 0], z_mean_tsne[:, 1],
    color="black",
    marker="x",
    s=100,
    label="Centroid"
)

plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-translation-vectors.png")