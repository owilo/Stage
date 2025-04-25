import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from sklearn.manifold import TSNE

from code.utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

digits = np.array([
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],   # 0
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217],    # 1
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032],    # 2
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787],    # 3
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937],  # 4
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428], # 5
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178],    # 6
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749],    # 7
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385],  # 8
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977]  # 9
])

autoencoder, _ = models.select_model(
    models.list_models(criteria={"type":"autoencoder", "labels" : False, "dataset_range" : (0, 1)})
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

y_src, y_dst = 2, 6

i_src = digits[y_src]

z_src = latent.encode(
    autoencoder,
    x_test[i_src],
    y_test[i_src],
    2,
    save_cache=True
)

z_dst = latent.translate(z_src, np.full(len(i_src), y_src), np.full(len(i_src), y_dst), z_class_distributions, use_std=False)

x_src = autoencoder.decoder.predict(z_src)
x_dst = autoencoder.decoder.predict(z_dst)

z_all, config = utils.pack_arrays(z_train, z_mean, z_src, z_dst)
tsne = TSNE(n_components=2, random_state=1337, perplexity=75)
z_tsne = tsne.fit_transform(z_all)
z_train_tsne, z_mean_tsne, z_src_tsne, z_dst_tsne = utils.unpack_arrays(z_tsne, config)

fig = plt.figure(figsize=(18, 10), constrained_layout=True)
gs  = gridspec.GridSpec(5, 9, figure=fig)

for row, idx in enumerate(np.arange(5)):
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(x_src[idx].squeeze(), cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[row, 1])
    ax.imshow(x_dst[idx].squeeze(), cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])

for row, idx in enumerate(np.arange(5, 10)):
    ax = fig.add_subplot(gs[row, 2])
    ax.imshow(x_src[idx].squeeze(), cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[row, 3])
    ax.imshow(x_dst[idx].squeeze(), cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])

ax_tsne = fig.add_subplot(gs[:, 4:9])
ax_tsne.set_aspect('equal', 'box')

ax_tsne.scatter(z_train_tsne[:, 0], z_train_tsne[:, 1], c=y_train, cmap="Paired", alpha=0.35, s=6)

classes = np.unique(y_test)
norm = Normalize(vmin=classes.min(), vmax=classes.max())
for c in classes:
    ax_tsne.scatter([], [], color=plt.cm.Paired(norm(c)), label=str(c))

ax_tsne.scatter(*z_mean_tsne[y_src], marker="x", c="red", s=100, label="Source centroid")
ax_tsne.scatter(*z_mean_tsne[y_dst], marker="x", c="blue", s=100, label="Destination centroid")

ax_tsne.arrow(
    *z_mean_tsne[y_src],
    *(z_mean_tsne[y_dst] - z_mean_tsne[y_src]),
    color="black",
    width=0.2,
    head_width=0.5,
    length_includes_head=True,
    label="Centroid translation"
)

ax_tsne.scatter(z_src_tsne[:,0], z_src_tsne[:,1], marker="+", c="red", s=150, label="Source sample")
ax_tsne.scatter(z_dst_tsne[:,0], z_dst_tsne[:,1], marker="+", c="blue", s=150, label="Destination sample")

for i in range(len(z_src_tsne)):
    ax_tsne.arrow(
        *z_src_tsne[i], *(z_dst_tsne[i] - z_src_tsne[i]),
        color="purple", width=0.2, head_width=0.5,
        length_includes_head=True,
        label="Sample translation" if i==0 else None
    )

ax_tsne.set_title(f"t-SNE : Translation from {y_src} → {y_dst}")
ax_tsne.legend(loc="upper left")
#plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / f"mnist-batch-translation-tsne-{y_src}-{y_dst}.png")