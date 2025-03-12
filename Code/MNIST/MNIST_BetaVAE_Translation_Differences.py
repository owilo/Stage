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

X_class_src = X_reencoded_train[Y_train == src_class]
X_class_dst = X_reencoded_train[Y_train == dst_class]

encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, 32)

encoded_means_src = encoded_means[src_class]
encoded_means_dst = encoded_means[dst_class]

encoded_std = utils.encoded_std(X_train, Y_train, "encoded_std_disvae", encoder, decoder, 2, 32)
encoded_std_src = encoded_std[src_class]
encoded_std_dst = encoded_std[dst_class]

sources = X_reencoded_valid[Y_valid == src_class]

translated = sources + encoded_means_dst - encoded_means_src
translated_std = encoded_means_dst + (encoded_std_dst / encoded_std_src) * (sources - encoded_means_src)


d = X_class_src.shape[1]

positions = 2 * np.arange(1, d + 1)

positions_src = positions - 0.35
positions_dst = positions + 0.35
positions_trans = positions - 0.125
positions_trans_std = positions + 0.125


plt.figure(figsize=(14, 8))
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

bp_trans_std = plt.boxplot(translated_std, positions=positions_trans_std, patch_artist=True, 
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
plt.savefig(f"./Results/mnist-translation-boxplot-{src_class}-{dst_class}.png")