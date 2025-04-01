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

src_digit = 1303
src_class = y_test[src_digit]
dst_class = 3

z_class_distributions = latent.encode_class_distributions(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

z_1 = latent.encode(autoencoder, x_test[src_digit : src_digit + 1], y_test)
x_1 = latent.decode(autoencoder, z_1, y_test)
z_2 = latent.encode(autoencoder, x_1, y_test)
x_2 = latent.decode(autoencoder, z_2, y_test)
z_3 = latent.encode(autoencoder, x_2, y_test)

z_mean_src = z_class_distributions[src_class][0]
z_mean_dst = z_class_distributions[dst_class][0]

translation = z_mean_dst - z_mean_src

z_trans_2 = z_2 + translation
z_trans_3 = z_3 + translation

x_trans_2 = latent.decode(autoencoder, z_trans_2, np.array([dst_class]))
x_trans_3 = latent.decode(autoencoder, z_trans_3, np.array([dst_class]))

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(x_test[src_digit].reshape(28, 28))
plt.axis("off")
plt.title("Source")

plt.subplot(1, 3, 2)
plt.imshow(x_trans_2[0].reshape(64, 64))
plt.axis("off")
plt.title("Décodé")

plt.subplot(1, 3, 3)
plt.imshow(x_trans_3[0].reshape(64, 64))
plt.axis("off")
plt.title("2x Décodé")

plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-decoded-{src_class}-{dst_class}.png")

fig, axes = plt.subplots(3, 2, figsize=(22, 18))
axes = axes.flatten()

axes[0].plot(z_mean_dst - z_mean_src,
             color="gray", ls="--", lw=0.75,
             label=f"Différence (translation {src_class} → {dst_class})")
axes[0].plot(z_mean_src,
             color="red", lw=2.25,
             label=f"Centroïde source ({src_class})")
axes[0].plot(z_mean_dst,
             color="blue", lw=2.25,
             label=f"Centroïde destination ({dst_class})")
axes[0].set_title("Centroïdes")
axes[0].legend(loc="lower left")

axes[1].plot(z_1.squeeze() - z_mean_src,
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde source {src_class})")
axes[1].plot(z_mean_src,
             color="red", ls="--", lw=1.5,
             label=f"Centroïde source ({src_class})")
axes[1].plot(z_1.squeeze(),
             color="darkred", lw=2.25,
             label=f"Chiffre source ({src_class})")
axes[1].set_title("Centroïde source et Chiffre source")
axes[1].legend(loc="lower left")

axes[2].plot(z_trans_3.squeeze() - z_mean_dst,
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde destination {dst_class})")
axes[2].plot(z_mean_dst,
             color="blue", ls="--", lw=1.5,
             label=f"Centroïde destination ({dst_class})")
axes[2].plot(z_trans_3.squeeze(),
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[2].set_title("Centroïde destination et Chiffre décodé & translaté")
axes[2].legend(loc="lower left")

axes[3].plot(z_trans_3.squeeze() - z_mean_src,
             color="gray", ls="--", lw=0.75,
             label=f"Différence (écart au centroïde source {src_class})")
axes[3].plot(z_mean_src,
             color="red", ls="--", lw=1.5,
             label=f"Centroïde source ({src_class})")
axes[3].plot(z_trans_3.squeeze(),
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[3].set_title("Centroïde source et Chiffre décodé & translaté")
axes[3].legend(loc="lower left")

axes[4].plot(z_trans_3.squeeze() - z_trans_2.squeeze(),
             color="gray", ls="--", lw=0.75,
             label="Différence")
axes[4].plot(z_trans_3.squeeze(),
             color="darkblue", lw=2.25,
             label=f"Chiffre décodé & translaté ({dst_class})")
axes[4].plot(z_trans_2.squeeze(),
             color="#32CD32", ls="--", lw=1.5,
             label=f"Chiffre translaté ({dst_class})")
axes[4].set_title("Chiffre translaté et Chiffre décodé & translaté")
axes[4].legend(loc="lower left")

fig.delaxes(axes[5])

for ax in axes[:5]:
    ax.grid(True, which="both")
    ax.axhline(y=0, color="gray")
    ax.set_xlabel("Indice")
    ax.set_ylabel("Valeur")

plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-plot-{src_class}-{dst_class}.png")
