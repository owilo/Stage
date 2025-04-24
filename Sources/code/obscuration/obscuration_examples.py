import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE

from code.utils import cache, latent, utils, models

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

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
    620, # 0
    3701, # 3
    9100, # 3
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

xp_src = autoencoder.decoder.predict(z_src)

key = 3

np.random.seed(key)
tf.keras.utils.set_random_seed(key)

u = np.random.randint(0, 9, digit_indices.shape[0])
y_dst = (u + y_src) % 10

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

z_class_distributions = latent.class_distributions(z_train, y_train)

sorted_keys = sorted(z_class_distributions)
z_mean = np.array([z_class_distributions[c][0] for c in sorted_keys])
z_std = np.array([z_class_distributions[c][1] for c in sorted_keys])

per_sample_std = z_std[y_src]
alpha = np.random.normal(0.0, per_sample_std)

z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions, use_std=False)

z_src_alpha = z_src + alpha
z_dst_alpha = latent.translate(z_src_alpha, y_src, y_dst, z_class_distributions, use_std=False)

x_dst = autoencoder.decoder.predict(z_dst)
x_dst_alpha = autoencoder.decoder.predict(z_dst_alpha)

_, _, z_invdst_alpha = autoencoder.encoder.predict(x_dst_alpha)

z_invdst = z_invdst_alpha - alpha
z_invsrc = latent.translate(z_invdst, y_dst, y_src, z_class_distributions, use_std=False)

x_invsrc = autoencoder.decoder.predict(z_invsrc)

fig, axes = plt.subplots(digit_indices.shape[0], 5, figsize=(10, 2 * digit_indices.shape[0]))

for i in range(digit_indices.shape[0]):
    axes[i, 0].imshow(x_src[i], cmap="gray")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(xp_src[i], cmap="gray")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(x_dst[i], cmap="gray")
    axes[i, 2].axis('off')

    axes[i, 3].imshow(x_dst_alpha[i], cmap="gray")
    axes[i, 3].axis('off')

    axes[i, 4].imshow(x_invsrc[i], cmap="gray")
    axes[i, 4].axis('off')

axes[0, 0].set_title("Original", fontsize=18)
axes[0, 1].set_title("Reencoded", fontsize=18)
axes[0, 2].set_title("Translated", fontsize=18)
axes[0, 3].set_title("Obscured", fontsize=18)
axes[0, 4].set_title("Recovered", fontsize=18)
    
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / f"mnist-obscuration-translation-examples-{key}.png")


z_all, configuration = utils.pack_arrays(z_train, z_mean, z_src, z_dst, z_dst_alpha)

tsne = TSNE(n_components=2, random_state=1337, perplexity=50)

z_tsne = tsne.fit_transform(z_all)

z_train_tsne, z_mean_tsne, z_src_tsne, z_dst_tsne, z_dst_alpha_tsne = utils.unpack_arrays(z_tsne, configuration)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    z_train_tsne[:, 0],
    z_train_tsne[:, 1],
    c=y_train,
    cmap="Paired",
    alpha=0.35,
    s=6
)

unique_classes = np.unique(y_test)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

plt.scatter(z_mean_tsne[:, 0], z_mean_tsne[:, 1], marker="x", color="black", s=100, label="Centroid")

plt.scatter(z_src_tsne[:, 0], z_src_tsne[:, 1], marker="+", color="red", s=150, label="Source latent")
plt.scatter(z_dst_alpha_tsne[:, 0], z_dst_alpha_tsne[:, 1], marker="+", color="blue", s=150, label="Target latent")

for i in range(len(z_src_tsne)):
    plt.plot(
        [z_src_tsne[i, 0], z_dst_tsne[i, 0]],
        [z_src_tsne[i, 1], z_dst_tsne[i, 1]],
        linestyle='--', color='purple', linewidth=1, label="Translation" if i == 0 else None
    )
    
    plt.plot(
        [z_dst_tsne[i, 0], z_dst_alpha_tsne[i, 0]],
        [z_dst_tsne[i, 1], z_dst_alpha_tsne[i, 1]],
        linestyle='--', color='magenta', linewidth=1, label="α perturbation" if i == 0 else None
    )

    plt.arrow(
        z_src_tsne[i, 0], z_src_tsne[i, 1],
        z_dst_alpha_tsne[i, 0] - z_src_tsne[i, 0],
        z_dst_alpha_tsne[i, 1] - z_src_tsne[i, 1],
        color='black', linewidth=1, head_width=2, head_length=2, length_includes_head=True, label="Perturbed translation" if i == 0 else None
    )

plt.title(f"t-SNE : Visualisation of the obscuration of various digits")
plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-obscuration-translation-tsne.png")