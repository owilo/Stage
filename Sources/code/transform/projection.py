import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from code.utils import cache, latent, utils, models

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

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

y_src = 2
y_dst = 6

z_test_src = latent.encode(autoencoder, x_test, y_test, 3, save_cache=True)
z_class_distributions = latent.encode_class_distributions(autoencoder, x_train, y_train, 2, save_cache=True)

source_classes = np.full(10, y_src)
destination_classes = np.full(10, y_dst)

z_translated = latent.translate(z_test_src[digits[y_src]], source_classes, destination_classes, z_class_distributions, False)

z_all = np.concatenate((
    z_test_src, 
    z_translated, 
    np.expand_dims(z_class_distributions[y_src][0], axis=0), 
    np.expand_dims(z_class_distributions[y_dst][0], axis=0)
))

# t-SNE
tsne = TSNE(n_components=2, random_state=1337, max_iter=300)
z_tsne = tsne.fit_transform(z_all)
z_tsne_test = z_tsne[:-12]
z_tsne_translated = z_tsne[-12:-2]
z_tsne_src_mean = z_tsne[-2]
z_tsne_dst_mean = z_tsne[-1]

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    z_tsne_test[:, 0],
    z_tsne_test[:, 1],
    c=y_test,
    cmap="Paired",
    alpha=0.35,
    s=6
)
unique_classes = np.unique(y_test)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))
plt.scatter(z_tsne_src_mean[0], z_tsne_src_mean[1], marker="x", color="red", s=100, label="Centroïde source")
plt.scatter(z_tsne_dst_mean[0], z_tsne_dst_mean[1], marker="x", color="blue", s=100, label="Centroïde destination")
plt.arrow(z_tsne_src_mean[0], z_tsne_src_mean[1],
          z_tsne_dst_mean[0] - z_tsne_src_mean[0],
          z_tsne_dst_mean[1] - z_tsne_src_mean[1],
          color="black", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (centres)")
for i in range(10):
    src = z_tsne_test[digits[y_src][i]]
    dst = z_tsne_translated[i]
    plt.scatter(src[0], src[1], marker="+", color="red", s=150)
    plt.scatter(dst[0], dst[1], marker="+", color="blue", s=150)
    plt.arrow(src[0], src[1],
              dst[0] - src[0],
              dst[1] - src[1],
              color="purple", width=0.01, head_width=0.2, length_includes_head=True)
plt.scatter([], [], marker="+", color="red", label="Chiffre source", s=150)
plt.scatter([], [], marker="+", color="blue", label="Chiffre translaté", s=150)
plt.arrow([], [], [], [], color="purple", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (chiffres)")
plt.title(f"t-SNE : Translation de {y_src} vers {y_dst}")
plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-translation-tsne.png")

# ACP 2D
pca2d = PCA(n_components=2, random_state=1337)
z_pca2d = pca2d.fit_transform(z_all)
z_pca2d_test = z_pca2d[:-12]
z_pca2d_translated = z_pca2d[-12:-2]
z_pca2d_src_mean = z_pca2d[-2]
z_pca2d_dst_mean = z_pca2d[-1]

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    z_pca2d_test[:, 0],
    z_pca2d_test[:, 1],
    c=y_test,
    cmap="Paired",
    alpha=0.35,
    s=6
)
unique_classes = np.unique(y_test)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))
plt.scatter(z_pca2d_src_mean[0], z_pca2d_src_mean[1], marker="x", color="red", s=100, label="Centroïde source")
plt.scatter(z_pca2d_dst_mean[0], z_pca2d_dst_mean[1], marker="x", color="blue", s=100, label="Centroïde destination")
plt.arrow(z_pca2d_src_mean[0], z_pca2d_src_mean[1],
          z_pca2d_dst_mean[0] - z_pca2d_src_mean[0],
          z_pca2d_dst_mean[1] - z_pca2d_src_mean[1],
          color="black", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (centres)")
for i in range(10):
    src = z_pca2d_test[digits[y_src][i]]
    dst = z_pca2d_translated[i]
    plt.scatter(src[0], src[1], marker="+", color="red", s=150)
    plt.scatter(dst[0], dst[1], marker="+", color="blue", s=150)
    plt.arrow(src[0], src[1],
              dst[0] - src[0],
              dst[1] - src[1],
              color="purple", width=0.01, head_width=0.2, length_includes_head=True)
plt.scatter([], [], marker="+", color="red", label="Chiffre source", s=150)
plt.scatter([], [], marker="+", color="blue", label="Chiffre translaté", s=150)
plt.arrow([], [], [], [], color="purple", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (chiffres)")
plt.title(f"ACP 2D : Translation de {y_src} vers {y_dst}")
plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-translation-pca2d.png")

# ACP 3D
pca3d = PCA(n_components=3, random_state=1337)
z_pca3d = pca3d.fit_transform(z_all)
z_pca3d_test = z_pca3d[:-12]
z_pca3d_translated = z_pca3d[-12:-2]
z_pca3d_src_mean = z_pca3d[-2]
z_pca3d_dst_mean = z_pca3d[-1]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z_pca3d_test[:, 0], z_pca3d_test[:, 1], z_pca3d_test[:, 2],
                c=y_test, cmap="Paired", alpha=0.35, s=6)
unique_classes = np.unique(y_test)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    ax.scatter([], [], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))
ax.scatter(z_pca3d_src_mean[0], z_pca3d_src_mean[1], z_pca3d_src_mean[2],
           marker="x", color="red", s=100, label="Centroïde source")
ax.scatter(z_pca3d_dst_mean[0], z_pca3d_dst_mean[1], z_pca3d_dst_mean[2],
           marker="x", color="blue", s=100, label="Centroïde destination")
dx = z_pca3d_dst_mean[0] - z_pca3d_src_mean[0]
dy = z_pca3d_dst_mean[1] - z_pca3d_src_mean[1]
dz = z_pca3d_dst_mean[2] - z_pca3d_src_mean[2]
ax.quiver(z_pca3d_src_mean[0], z_pca3d_src_mean[1], z_pca3d_src_mean[2],
          dx, dy, dz, color="black", arrow_length_ratio=0.1, label="Translation (centres)")
for i in range(10):
    src = z_pca3d_test[digits[y_src][i]]
    dst = z_pca3d_translated[i]
    ax.scatter(src[0], src[1], src[2], marker="+", color="red", s=150)
    ax.scatter(dst[0], dst[1], dst[2], marker="+", color="blue", s=150)
    ax.quiver(src[0], src[1], src[2],
              dst[0] - src[0], dst[1] - src[1], dst[2] - src[2],
              color="purple", arrow_length_ratio=0.1)
ax.legend()
plt.title(f"ACP 3D : Translation de {y_src} vers {y_dst}")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-translation-pca3d.png")

# LDA
labels_all = np.concatenate((y_test, np.full(10, y_dst), np.array([y_src, y_dst])))

lda = LDA(n_components=2)
z_lda = lda.fit_transform(z_all, labels_all)
z_lda_test = z_lda[:-12]
z_lda_translated = z_lda[-12:-2]
z_lda_src_mean = z_lda[-2]
z_lda_dst_mean = z_lda[-1]

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    z_lda_test[:, 0],
    z_lda_test[:, 1],
    c=y_test,
    cmap="Paired",
    alpha=0.35,
    s=6
)
unique_classes = np.unique(y_test)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))
plt.scatter(z_lda_src_mean[0], z_lda_src_mean[1], marker="x", color="red", s=100, label="Centroïde source")
plt.scatter(z_lda_dst_mean[0], z_lda_dst_mean[1], marker="x", color="blue", s=100, label="Centroïde destination")
plt.arrow(z_lda_src_mean[0], z_lda_src_mean[1],
          z_lda_dst_mean[0] - z_lda_src_mean[0],
          z_lda_dst_mean[1] - z_lda_src_mean[1],
          color="black", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (centres)")
for i in range(10):
    src = z_lda_test[digits[y_src][i]]
    dst = z_lda_translated[i]
    plt.scatter(src[0], src[1], marker="+", color="red", s=150)
    plt.scatter(dst[0], dst[1], marker="+", color="blue", s=150)
    plt.arrow(src[0], src[1],
              dst[0] - src[0],
              dst[1] - src[1],
              color="purple", width=0.01, head_width=0.2, length_includes_head=True)
plt.scatter([], [], marker="+", color="red", label="Chiffre source", s=150)
plt.scatter([], [], marker="+", color="blue", label="Chiffre translaté", s=150)
plt.arrow([], [], [], [], color="purple", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (chiffres)")
plt.title(f"LDA : Translation de {y_src} vers {y_dst}")
plt.legend()
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Projections" / "mnist-translation-lda.png")