import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_selection import f_classif

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

F_scores, p_values = f_classif(X_reencoded_train, Y_train)

plt.figure(figsize=(10, 8))
plt.bar(range(X_reencoded_train.shape[1]), F_scores, color='skyblue')
plt.xlabel("Dimension")
plt.ylabel("ANOVA F-score")
plt.title("Pouvoir discriminant des dimensions latentes")
plt.xticks(range(X_reencoded_train.shape[1]))
plt.savefig("./Results/mnist-anova.png")
