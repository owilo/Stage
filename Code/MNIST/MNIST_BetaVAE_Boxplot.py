import numpy as np
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

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-16-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-16-decoder.keras")

X_reencoded_train = utils.encoded(X_train, "train_disvae", encoder, decoder, 2, batch_size)

for i in range(10):
    X_class = X_reencoded_train[Y_train == i]

    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='gray')

    plt.boxplot(X_class, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red', linewidth=3),
            flierprops=dict(markerfacecolor='orange', marker='o', markersize=8, alpha=0.25, markeredgewidth=0))
    
    plt.xlabel("Dimension")
    plt.ylabel("Valeur")
    plt.title(f"Centroïde de la classe {i}")
    plt.tight_layout()
    plt.savefig(f"./Results/Centroid-Boxplots/mnist-centroid-boxplot-{i}.png")