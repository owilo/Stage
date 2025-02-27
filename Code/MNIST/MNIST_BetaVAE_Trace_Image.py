import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.manifold import TSNE

import utils

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train64 = tf.image.resize(X_train, (64, 64))
X_valid64 = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

X_reencoded_valid = utils.encoded(X_valid64, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train64, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

digits = [
    [157, 713, 1261, 3911, 5684, 5865, 8067, 8199, 8681, 9753],  # 0
    [31, 783, 1240, 2719, 4308, 4428, 4759, 6202, 6308, 7217],  # 1
    [291, 741, 888, 1210, 1303, 2253, 4445, 5407, 7977, 9032],  # 2
    [614, 865, 923, 2881, 3493, 3686, 4925, 7329, 8598, 9787],  # 3
    [117, 1059, 1849, 2307, 4813, 5525, 5559, 6516, 7669, 7937],  # 4
    [1089, 2525, 3788, 4094, 4196, 5445, 5364, 7475, 8122, 9428],  # 5
    [54, 164, 1108, 2483, 2766, 2876, 6842, 8200, 8828, 9178],  # 6
    [410, 522, 880, 1750, 4073, 4467, 5205, 6079, 6380, 8749],  # 7
    [914, 2004, 2451, 4165, 6297, 7313, 7713, 8466, 9042, 9385],  # 8
    [1869, 3840, 4843, 5456, 7246, 7382, 8084, 8372, 8899, 8977]  # 9
]

src_class = 2
dst_class = 7
translation = encoded_means[dst_class] - encoded_means[src_class]
X_src = X_reencoded_valid[Y_valid == src_class]
translated = X_src + translation
translated_minus_source = translated - encoded_means[src_class]
source_minus_translated = encoded_means[src_class] - translated
translated_minus_dest = translated - encoded_means[dst_class]
dest_minus_translated = encoded_means[dst_class] - translated

fig, axes = plt.subplots(6, 10, figsize=(25, 12))

axes[0, 0].set_ylabel("Source", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[1, 0].set_ylabel("Translaté", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[2, 0].set_ylabel("Trans. - Proto source", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[3, 0].set_ylabel("Proto source - Trans.", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[4, 0].set_ylabel("Trans. - Proto dest.", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[5, 0].set_ylabel("Proto dest. - Trans.", rotation=0, fontsize=26, labelpad=20, ha="right")

for i in range(10):
    mask = (Y_valid == src_class)
    indices = np.where(mask)[0]
    digit = np.where(indices == digits[src_class][i])[0][0]

    axes[0, i].imshow(decoder.predict(np.expand_dims(X_src[digit], axis=0))[0], cmap="gray")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(decoder.predict(np.expand_dims(translated[digit], axis=0))[0], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

    axes[2, i].imshow(decoder.predict(np.expand_dims(translated_minus_source[digit], axis=0))[0], cmap="gray")
    axes[2, i].set_xticks([])
    axes[2, i].set_yticks([])
    
    axes[3, i].imshow(decoder.predict(np.expand_dims(source_minus_translated[digit], axis=0))[0], cmap="gray")
    axes[3, i].set_xticks([])
    axes[3, i].set_yticks([])

    axes[4, i].imshow(decoder.predict(np.expand_dims(translated_minus_dest[digit], axis=0))[0], cmap="gray")
    axes[4, i].set_xticks([])
    axes[4, i].set_yticks([])
    
    axes[5, i].imshow(decoder.predict(np.expand_dims(dest_minus_translated[digit], axis=0))[0], cmap="gray")
    axes[5, i].set_xticks([])
    axes[5, i].set_yticks([])

plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-images{src_class}{dst_class}.png")

data = np.concatenate([
    X_src,
    translated,
    translated_minus_source,
    source_minus_translated,
    translated_minus_dest,
    dest_minus_translated
], axis=0)

n_points = X_src.shape[0]
labels = ([f"Source ({src_class})"] * n_points +
          [f"Translaté ({dst_class})"] * n_points +
          ["Trans. - Proto source"] * n_points +
          ["Proto source - Trans."] * n_points +
          ["Trans. - Proto dest."] * n_points +
          ["Proto dest. - Trans."] * n_points)

tsne = TSNE(n_components=2, random_state=1337, max_iter=300)
data_tsne = tsne.fit_transform(data)

unique_labels = np.unique(labels)
n_labels = len(unique_labels)

cmap = plt.get_cmap('Paired')
colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

plt.figure(figsize=(8, 8))
for label in unique_labels:
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(data_tsne[idx, 0], data_tsne[idx, 1], color=colors[label], label=label, alpha=0.35)

plt.legend()
plt.title(f"t-SNE de détection de traces (translation de {src_class} vers {dst_class})")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"./Results/mnist-trace-images-tsne-{src_class}{dst_class}.png")