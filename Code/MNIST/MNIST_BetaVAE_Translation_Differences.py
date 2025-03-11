import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-16-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-16-decoder.keras")

X_reencoded_train = utils.encoded(X_train, "train_disvae", encoder, decoder, 2, batch_size)
X_reencoded_valid = utils.encoded(X_valid, "test_disvae", encoder, decoder, 2, batch_size)

src_class = 0
dst_class = 1

encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, 32)

X_class_src = X_reencoded_train[Y_train == src_class]
X_class_dst = X_reencoded_train[Y_train == dst_class]

encoded_means_src = encoded_means[src_class]
encoded_means_dst = encoded_means[dst_class]
translation = encoded_means_dst - encoded_means_src
translated = X_reencoded_valid[Y_valid == src_class] + translation

d = X_class_src.shape[1]

positions_dst = 2 * np.arange(1, d + 1)
positions_src = positions_dst - 0.25
positions_trans = positions_dst + 0.25

plt.figure(figsize=(16, 8))
plt.axhline(y=0, color='gray')

bp_src = plt.boxplot(X_class_src, positions=positions_src, patch_artist=True, 
                     showfliers=False, widths=0.2,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='darkblue', linewidth=3))

bp_dst = plt.boxplot(X_class_dst, positions=positions_dst, patch_artist=True, 
                     showfliers=False, widths=0.2,
                     boxprops=dict(facecolor='lightgreen', color='green'),
                     medianprops=dict(color='darkgreen', linewidth=3))

bp_trans = plt.boxplot(translated, positions=positions_trans, patch_artist=True, 
                       showfliers=False, widths=0.2,
                       boxprops=dict(facecolor='lightcoral', color='red'),
                       medianprops=dict(color='darkred', linewidth=3))

plt.xlabel("Dimension")
plt.ylabel("Valeur")
plt.title("Translation")

plt.xticks(2 * np.arange(1, d + 1), [f"{i}" for i in range(1, d + 1)])

legend_handles = [
    mpatches.Patch(color='lightblue', label='Source'),
    mpatches.Patch(color='lightgreen', label='Destination'),
    mpatches.Patch(color='lightcoral', label='Translation')
]
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-boxplot-{src_class}-{dst_class}.png")