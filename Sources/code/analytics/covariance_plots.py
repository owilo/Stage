import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist

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

cov_whole = np.cov(z_train, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_whole, cmap="PiYG", center=0.0, vmin=-2, vmax=2)
plt.title("Covariance")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "CovariancePlots" / "covariance-plot.png")
plt.close()

for i in range(10):
    z_class = z_train[y_train == i]
    
    cov_class = np.cov(z_class, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_class, cmap="PiYG", center=0.0, vmin=-2, vmax=2)
    plt.title(f"Covariance classe {i}")
    plt.tight_layout()
    plt.savefig(cache.RESULTS_FOLDER / "CovariancePlots" / f"covariance-plot-{i}.png")
    plt.close()