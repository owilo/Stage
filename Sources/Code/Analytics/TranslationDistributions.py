import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Code.Utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = models.select_model(models.list_models(
    criteria={"type": "VAE"},
    formatter=models.AE_FORMATTER
))

y_src = 0
y_dst = 1

x_selected = x_train[(y_train == y_src) | (y_train == y_dst)]
y_selected = y_train[(y_train == y_src) | (y_train == y_dst)]

z_selected = latent.encode(
    autoencoder,
    x=x_selected,
    y=y_selected,
    n_times=2,
    save_cache=False
)

z_test = latent.encode(
    autoencoder,
    x=x_test,
    y=y_test,
    n_times=3,
    save_cache=True
)

z_class_distributions = latent.class_distributions(z_selected, y_selected)

z_mean_src, z_std_src = z_class_distributions[y_src]
z_mean_dst, z_std_dst = z_class_distributions[y_dst]

z_src = z_test[y_test == y_src]
z_dst = z_test[y_test == y_dst]

z_trans = latent.translate(z_src, y_src, y_dst, z_class_distributions, use_std=False)
z_trans_std = latent.translate(z_src, y_src, y_dst, z_class_distributions, use_std=True)

d = z_test.shape[1]

positions = 2 * np.arange(1, d + 1)

positions_src = positions - 0.35
positions_dst = positions + 0.35
positions_trans = positions - 0.125
positions_trans_std = positions + 0.125


plt.figure(figsize=(14, 8))
plt.axhline(y=0, color='gray')

bp_src = plt.boxplot(z_src, positions=positions_src, patch_artist=True, 
                     showfliers=False, widths=0.2,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='darkblue', linewidth=3))

bp_dst = plt.boxplot(z_dst, positions=positions_dst, patch_artist=True, 
                     showfliers=False, widths=0.2,
                     boxprops=dict(facecolor='lightgreen', color='green'),
                     medianprops=dict(color='darkgreen', linewidth=3))

bp_trans = plt.boxplot(z_trans, positions=positions_trans, patch_artist=True, 
                       showfliers=False, widths=0.2,
                       boxprops=dict(facecolor='lightcoral', color='red'),
                       medianprops=dict(color='darkred', linewidth=3))

bp_trans_std = plt.boxplot(z_trans_std, positions=positions_trans_std, patch_artist=True, 
                       showfliers=False, widths=0.2,
                       boxprops=dict(facecolor='mediumpurple', color='purple'),
                       medianprops=dict(color='darkviolet', linewidth=3))

plt.xlabel("Dimension")
plt.ylabel("Valeur")
plt.title("Translation")

plt.xticks(2 * np.arange(1, d + 1), [f"{i}" for i in range(1, d + 1)])

legend_handles = [
    mpatches.Patch(color='lightblue', label='Source'),
    mpatches.Patch(color='lightcoral', label='Translaté'),
    mpatches.Patch(color='mediumpurple', label='Translaté & Normalisé'),
    mpatches.Patch(color='lightgreen', label='Destination')
]
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / f"mnist-translation-boxplot-{y_src}-{y_dst}.png")