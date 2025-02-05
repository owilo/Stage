import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

import utils

np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_reencoded_all = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

digits = [
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],  # 0
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217], # 1
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032], # 2
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787], # 3
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937], # 4
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428], # 5
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178], # 6
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749], # 7
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385], # 8
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977] # 9
]

src_class = 1
dst_class = 3

mean_encoded_src = encoded_means[src_class]
mean_encoded_dst = encoded_means[dst_class]
translation = mean_encoded_dst - mean_encoded_src

X_encoded = X_reencoded_all[digits[src_class]]
translated = X_encoded + translation

X_reencoded_all = np.concatenate((
    X_reencoded_all,
    translated,
    np.expand_dims(mean_encoded_src, 0),
    np.expand_dims(mean_encoded_dst, 0)
))

pca = PCA(n_components=3, random_state=1337)
X_pca = pca.fit_transform(X_reencoded_all)

X_pca_original = X_pca[:-12]
X_pca_translated = X_pca[-12:-2]
X_pca_src_centroid = X_pca[-2]
X_pca_dst_centroid = X_pca[-1]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca_original[:, 0],
    X_pca_original[:, 1],
    X_pca_original[:, 2],
    c=Y_valid,
    cmap="Paired",
    alpha=0.35,
    s=6
)

unique_classes = np.unique(Y_valid)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
for class_label in unique_classes:
    ax.scatter([], [], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

ax.scatter(X_pca_src_centroid[0], X_pca_src_centroid[1], X_pca_src_centroid[2],
           marker="x", color="red", s=100, label="Centroïde source")
ax.scatter(X_pca_dst_centroid[0], X_pca_dst_centroid[1], X_pca_dst_centroid[2],
           marker="x", color="blue", s=100, label="Centroïde destination")

dx = X_pca_dst_centroid[0] - X_pca_src_centroid[0]
dy = X_pca_dst_centroid[1] - X_pca_src_centroid[1]
dz = X_pca_dst_centroid[2] - X_pca_src_centroid[2]
ax.quiver(X_pca_src_centroid[0], X_pca_src_centroid[1], X_pca_src_centroid[2],
          dx, dy, dz, arrow_length_ratio=0.1, color="black", linewidth=1, label="Translation (centres)")

for i in range(10):
    original_idx = digits[src_class][i]
    src_point = X_pca_original[original_idx]
    tgt_point = X_pca_translated[i]
    
    ax.scatter(src_point[0], src_point[1], src_point[2],
               marker="+", color="red", s=100)
    ax.scatter(tgt_point[0], tgt_point[1], tgt_point[2],
               marker="+", color="blue", s=100)
    
    ax.quiver(src_point[0], src_point[1], src_point[2],
              tgt_point[0] - src_point[0],
              tgt_point[1] - src_point[1],
              tgt_point[2] - src_point[2],
              arrow_length_ratio=0.1, color="purple", linewidth=1)

ax.legend()
ax.set_title(f"PCA (3D): Translation de {src_class} vers {dst_class}")

plt.tight_layout()
plt.savefig("./Results/mnist-translation-pca-3d.png")
plt.show()