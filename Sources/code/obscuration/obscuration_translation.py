import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

from code.utils import cache, latent, utils, models

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

digit_indices = np.array([
    1333, # 0
    9415, # 1
    3773, # 2
     524, # 3
    1980, # 4
    1874, # 5
    4252, # 6
    6960, # 7
    8466, # 8
    5333  # 9
])

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))

# Original
# Translatés
# Translatés + perturbation
# Inv translatés

digit_indices = np.array([
    1333, # 0
    9415, # 1
    3773, # 2
     524, # 3
    1980, # 4
    1874, # 5
    4252, # 6
    6960, # 7
    8466, # 8
    5333  # 9
])

x_src = x_test[digit_indices]
y_src = y_test[digit_indices]

z_src = latent.encode(
    autoencoder,
    x=x_src,
    y=y_src,
    n_times=2,
    save_cache=False
)

fig, axes = plt.subplots(4, 10, figsize=(20, 8))

key = 2

np.random.seed(key)
tf.keras.utils.set_random_seed(key)

u = np.random.randint(0, 9, 10)
y_dst = (u + y_src) % 10

z_class_distributions = latent.encode_class_distributions(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

z_std = np.array([z_class_distributions[c][1] for c in sorted(z_class_distributions)])

alpha = np.random.normal(np.zeros_like(z_std), z_std)

z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions, use_std=False)

z_src_alpha = z_src + alpha
z_dst_alpha = latent.translate(z_src_alpha, y_src, y_dst, z_class_distributions, use_std=False)

x_dst = autoencoder.decoder.predict(z_dst)
x_dst_alpha = autoencoder.decoder.predict(z_dst_alpha)

_, _, z_invdst_alpha = autoencoder.encoder.predict(x_dst_alpha)

z_invdst = z_invdst_alpha - alpha
z_invsrc = latent.translate(z_invdst, y_dst, y_src, z_class_distributions, use_std=False)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

for i in range(10):
    axes[0, i].imshow(x_src[i], cmap="gray")
    axes[0, i].axis('off')

    axes[1, i].imshow(x_dst[i], cmap="gray")
    axes[1, i].axis('off')

    axes[2, i].imshow(x_dst_alpha[i], cmap="gray")
    axes[2, i].axis('off')

    axes[3, i].imshow(x_invsrc[i], cmap="gray")
    axes[3, i].axis('off')
    
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / f"mnist-obscuration-translation-{key}.png")