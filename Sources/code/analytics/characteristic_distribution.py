import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.stats import norm, anderson

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

z0 = z_train[:, 0]

unique_classes = np.unique(y_train)

total_plots = 1 + len(unique_classes)
rows = 3
cols = 4

fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
axs = axs.flatten()

ax = axs[0]
data = z0
mu, std = np.mean(data), np.std(data)
count, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, 'k', linewidth=2)
anderson_statistic = anderson(data, 'norm')
ax.set_title(f"All Data\nAnderson stat: {anderson_statistic.statistic:.3f}")

for idx, c in enumerate(unique_classes):
    ax = axs[idx + 1]
    data_class = z0[y_train == c]
    mu_c, std_c = np.mean(data_class), np.std(data_class)
    count, bins, _ = ax.hist(data_class, bins=30, density=True, alpha=0.6, color='b')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_c, std_c)
    ax.plot(x, p, 'r', linewidth=2)
    anderson_statistic = anderson(data_class, 'norm')
    ax.set_title(f"Classe {c}\nAnderson: {anderson_statistic.statistic:.3f}")

for i in range(total_plots, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()