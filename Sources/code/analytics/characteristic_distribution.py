import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import scipy.stats

from code.utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

dim = 0
z = z_train[:, dim]
unique_classes = np.unique(y_train)

total_plots = 1 + len(unique_classes)
rows = 3
cols = 4
fig, axs = plt.subplots(rows, cols, figsize=(12, 10))
axs = axs.flatten()

ax = axs[0]
data = z
mu, std = np.mean(data), np.std(data)
count, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'k', linewidth=2)
ks_statistic, ks_pvalue = scipy.stats.kstest(data, 'norm', args=(mu, std))
ax.set_title(f"Dataset\nKS: {ks_statistic:.3f} (p={ks_pvalue:.3f})\n")

for idx, c in enumerate(unique_classes):
    ax = axs[idx + 1]
    data_class = z[y_train == c]
    mu_c, std_c = np.mean(data_class), np.std(data_class)
    count, bins, _ = ax.hist(data_class, bins=30, density=True, alpha=0.6, color='b')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu_c, std_c)
    ax.plot(x, p, 'r', linewidth=2)
    ks_statistic, ks_pvalue = scipy.stats.kstest(data_class, 'norm', args=(mu_c, std_c))
    ax.set_title(f"Classe {c}\nKS: {ks_statistic:.3f} (p={ks_pvalue:.3f})\n")

for i in range(total_plots, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CharacteristicsDistributions" / f"dimension-distribution-{dim}.png")

n_dims = z_train.shape[1]
ks_scores = []
ks_pvalues = []
for i in range(n_dims):
    data = z_train[:, i]
    mu, std = np.mean(data), np.std(data)
    ks_stat, ks_p = scipy.stats.kstest(data, 'norm', args=(mu, std))
    ks_scores.append(ks_stat)
    ks_pvalues.append(ks_p)

fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8))

axs1[0].bar(range(n_dims), ks_scores, color='skyblue')
axs1[0].set_title("Score KS par dimension latente")
axs1[0].set_xlabel("Dimension latente")
axs1[0].set_ylabel("Score KS")
axs1[0].set(xlim=(-1, n_dims))

axs1[1].bar(range(n_dims), ks_pvalues, color='salmon')
axs1[1].axhline(0.05, color='red', linestyle='--', label='Seuil (0.05)')
axs1[1].set_title("Valeur p par dimension latente")
axs1[1].set_xlabel("Dimension latente")
axs1[1].set_ylabel("Valeur p")
axs1[1].set(xlim=(-1, n_dims))
axs1[1].legend()

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CharacteristicsDistributions" / "dimension-ks.png")

avg_ks_all = np.mean(ks_scores)
avg_p_all = np.mean(ks_pvalues)

avg_ks_classes = []
avg_p_classes = []
for c in unique_classes:
    indices = np.where(y_train == c)[0]
    ks_scores_c = []
    ks_pvalues_c = []
    for i in range(n_dims):
        data = z_train[indices, i]
        mu, std = np.mean(data), np.std(data)
        ks_stat, ks_p = scipy.stats.kstest(data, 'norm', args=(mu, std))
        ks_scores_c.append(ks_stat)
        ks_pvalues_c.append(ks_p)
    avg_ks_classes.append(np.mean(ks_scores_c))
    avg_p_classes.append(np.mean(ks_pvalues_c))

labels = ['Dataset'] + [f"Classe {c}" for c in unique_classes]
avg_ks_groups = [avg_ks_all] + avg_ks_classes
avg_p_groups = [avg_p_all] + avg_p_classes

fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))

axs2[0].bar(labels, avg_ks_groups, color='skyblue')
axs2[0].set_title("Score KS moyen par groupe")
axs2[0].set_xlabel("Groupe")
axs2[0].set_ylabel("Score KS moyen")
axs2[0].set(xlim=(-1, len(labels)))

axs2[1].bar(labels, avg_p_groups, color='salmon')
axs2[1].axhline(0.05, color='red', linestyle='--', label='Seuil (0.05)')
axs2[1].set_title("Valeur p moyenne par groupe")
axs2[1].set_xlabel("Groupe")
axs2[1].set_ylabel("Valeur p moyenne")
axs2[1].set(xlim=(-1, len(labels)))
axs2[1].legend()

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CharacteristicsDistributions" / "class-ks.png")