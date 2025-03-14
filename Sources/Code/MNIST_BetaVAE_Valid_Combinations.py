import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

import Utils

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)
X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-16-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-16-decoder.keras")

X_reencoded_train = Utils.encoded(X_train, "train_disvae", encoder, decoder, 2, batch_size)
X_reencoded_valid = Utils.encoded(X_valid, "test_disvae", encoder, decoder, 2, batch_size)

src_class = 0
dst_class = 1

X_class_src = X_reencoded_train[Y_train == src_class]
X_class_dst = X_reencoded_train[Y_train == dst_class]

z = X_class_dst
z0 = z[:, 0]

q1, q3 = np.percentile(z0, [25, 75])

bin_edges = np.linspace(q1, q3, num=6)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

conditional_means = {i: [] for i in range(1, z.shape[1])}

for i in range(len(bin_edges) - 1):
    lower, upper = bin_edges[i], bin_edges[i + 1]
    mask = (z0 >= lower) & (z0 < upper)
    z_bin = z[mask]
    for dim in range(1, z.shape[1]):
        if z_bin.shape[0] > 0:
            conditional_means[dim].append(np.mean(z_bin[:, dim]))
        else:
            conditional_means[dim].append(np.nan)

latent_profiles = np.zeros((len(bin_centers), z.shape[1]))
for i, center in enumerate(bin_centers):
    latent_profiles[i, 0] = center
    for dim in range(1, z.shape[1]):
        latent_profiles[i, dim] = conditional_means[dim][i]

fig, ax = plt.subplots(figsize=(10, 6))

dest_data = [X_class_dst[:, i] for i in range(16)]
bp_dst = ax.boxplot(dest_data, positions=np.arange(16), patch_artist=True, widths=0.5,
                    showfliers=False,
                    boxprops=dict(facecolor='lightgreen', color='green'),
                    medianprops=dict(color='darkgreen', linewidth=2),
                    whiskerprops=dict(color='green'),
                    capprops=dict(color='green'))

for element in bp_dst['boxes']:
    element.set_zorder(1)
for element in bp_dst['whiskers']:
    element.set_zorder(1)
for element in bp_dst['caps']:
    element.set_zorder(1)
for element in bp_dst['medians']:
    element.set_zorder(1)

x_ticks = np.arange(z.shape[1])
labels = [f"z{i}" for i in range(z.shape[1])]

colors = plt.cm.viridis(np.linspace(0, 1, len(bin_centers)))
for i in range(len(bin_centers)):
    ax.plot(x_ticks, latent_profiles[i, :], marker='o', color=colors[i],
            label=f'z0 = {bin_centers[i]:.2f}', zorder=10)

ax.set_xticks(x_ticks)
ax.set_xticklabels(labels)
ax.set_xlabel("Dimension latente")
ax.set_ylabel("Valeur")
ax.set_title("Évolution de la moyenne des dimensions latentes en fonction de z0")
ax.legend(title="z0 fixé", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("./Results/mnist-trace-combinations.png")