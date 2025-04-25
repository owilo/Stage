import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE

from code.utils import cache, latent, utils, models

(x_train, y_train), (_, _) = mnist.load_data()
x_train, = utils.preprocess_dataset(x_train)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

z_class_distributions = latent.class_distributions(z_train, y_train)
z_mean = np.array([z_class_distributions[c][0] for c in z_class_distributions])

z_all, configuration = utils.pack_arrays(z_train, z_mean)

tsne = TSNE(n_components=2, random_state=1337, perplexity=50)
z_tsne = tsne.fit_transform(z_all)

z_train_tsne, z_mean_tsne = utils.unpack_arrays(z_tsne, configuration)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    z_train_tsne[:, 0],
    z_train_tsne[:, 1],
    c=y_train,
    cmap="Paired",
    alpha=0.35,
    s=6
)

unique_classes = np.unique(y_train)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

plt.scatter(z_mean_tsne[:, 0], z_mean_tsne[:, 1], marker="x", color="black", s=100, label="Centroid")

plt.title(f"t-SNE : Visualisation of class clusters and centroids")
plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-centroids-tsne.png")