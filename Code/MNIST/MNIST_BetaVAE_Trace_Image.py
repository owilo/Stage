import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

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

src_class = 0
dst_class = 1
translation = encoded_means[dst_class] - encoded_means[src_class]

fig, axes = plt.subplots(6, 10, figsize=(25, 12))

axes[0, 0].set_ylabel("Source", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[1, 0].set_ylabel("Translaté", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[2, 0].set_ylabel("Trans. - Proto source", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[3, 0].set_ylabel("Proto source - Trans.", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[4, 0].set_ylabel("Trans. - Proto dest.", rotation=0, fontsize=26, labelpad=20, ha="right")
axes[5, 0].set_ylabel("Proto dest. - Trans.", rotation=0, fontsize=26, labelpad=20, ha="right")

for i in range(10):
    digit = digits[src_class][i]
    X_digit = np.expand_dims(X_reencoded_valid[digit], axis=0)

    axes[0, i].imshow(decoder.predict(X_digit)[0], cmap="gray")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    translated = X_digit + translation

    axes[1, i].imshow(decoder.predict(translated)[0], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

    translated_minus_source = translated - encoded_means[src_class]
    axes[2, i].imshow(decoder.predict(translated_minus_source)[0], cmap="gray")
    axes[2, i].set_xticks([])
    axes[2, i].set_yticks([])

    source_minus_translated = encoded_means[src_class] - translated
    axes[3, i].imshow(decoder.predict(source_minus_translated)[0], cmap="gray")
    axes[3, i].set_xticks([])
    axes[3, i].set_yticks([])

    translated_minus_dest = translated - encoded_means[dst_class]
    axes[4, i].imshow(decoder.predict(translated_minus_dest)[0], cmap="gray")
    axes[4, i].set_xticks([])
    axes[4, i].set_yticks([])

    dest_minus_translated = encoded_means[dst_class] - translated
    axes[5, i].imshow(decoder.predict(dest_minus_translated)[0], cmap="gray")
    axes[5, i].set_xticks([])
    axes[5, i].set_yticks([])

plt.tight_layout()
plt.savefig("./Results/mnist-trace-images.png")
