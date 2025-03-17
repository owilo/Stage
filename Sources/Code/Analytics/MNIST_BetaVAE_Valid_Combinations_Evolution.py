import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

from Utils import utils

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

encoder = load_model("../Models/DISVAE/mnist-16-encoder.keras")
decoder = load_model("../Models/DISVAE/mnist-16-decoder.keras")

X_reencoded_train = utils.encoded(X_train, "train_disvae", encoder, decoder, 2, batch_size)
X_reencoded_valid = utils.encoded(X_valid, "test_disvae", encoder, decoder, 2, batch_size)

src_class = 0
dst_class = 1

X_class_src = X_reencoded_train[Y_train == src_class]
X_class_dst = X_reencoded_train[Y_train == dst_class]

z = X_class_dst
z0 = z[:, 0]

q1, q3 = np.percentile(z0, [25, 75])

bin_edges = np.linspace(q1, q3, num=6)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

conditional_means = {i: [] for i in range(1, z.shape[1])}

for i in range(len(bin_edges) - 1):
    lower, upper = bin_edges[i], bin_edges[i + 1]
    mask = (z0 >= lower) & (z0 < upper)
    z_bin = z[mask]
    for dim in range(1, z.shape[1]):
        if z_bin.shape[0] > 0:
            conditional_means[dim].append(np.mean(z_bin[:, dim]))
        else:
            conditional_means[dim].append(np.nan)

cmap = plt.cm.tab20
colors = [cmap(i) for i in range(16)]

plt.figure(figsize=(10, 6))
for dim in range(1, z.shape[1]):
    plt.plot(bin_centers, conditional_means[dim], marker='o', label=f'z{dim}', color=colors[dim - 1])

plt.xlabel("z0 fixé")
plt.ylabel("Moyenne de la dimension latente")
plt.title("Évolution de la moyenne des dimensions latentes en fonction de z0")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("../Results/mnist-trace-combinations-evolution.png")