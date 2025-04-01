import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

from code.models import betaVAE
from code.utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae16.keras")

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
    n_times=3,
    save_cache=True
)

src_class = 0
dst_class = 1

z_class_src = z_train[y_train == src_class]
z_class_dst = z_train[y_train == dst_class]

z = z_class_dst
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

dest_data = [z_class_dst[:, i] for i in range(16)]
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
ax.set_title(f"Évolution de la moyenne des dimensions latentes en fonction de z0 ({src_class}→{dst_class})")
ax.legend(title="z0 fixé", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CharacteristicsInterdependancy" / "mnist-interdependancy.png")

cmap = plt.cm.tab20
colors = [cmap(i) for i in range(16)]

plt.figure(figsize=(10, 6))
for dim in range(1, z.shape[1]):
    plt.plot(bin_centers, conditional_means[dim], marker='o', label=f'z{dim}', color=colors[dim - 1])

plt.xlabel("z0 fixé")
plt.ylabel("Moyenne de la dimension latente")
plt.title(f"Évolution de la moyenne des dimensions latentes en fonction de z0 ({src_class}→{dst_class})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CharacteristicsInterdependancy" / "mnist-interdependancy-evolution.png")