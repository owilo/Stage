import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

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

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_encoded_all = encoder.predict(X_valid, batch_size=batch_size)
X_decoded_all = decoder.predict(X_encoded_all, batch_size=batch_size)
X_reencoded_all = encoder.predict(X_decoded_all, batch_size=batch_size)

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

mean_encoded_src = np.mean(X_reencoded_all[Y_valid == src_class], axis=0)
mean_encoded_dst = np.mean(X_reencoded_all[Y_valid == dst_class], axis=0)
translation = mean_encoded_dst - mean_encoded_src

X_encoded = X_reencoded_all[digits[src_class]]

translated = X_encoded + translation

X_reencoded_all = np.concatenate((X_reencoded_all, translated, 
                                    np.expand_dims(mean_encoded_src, 0), 
                                    np.expand_dims(mean_encoded_dst, 0)))

X_for_lda = X_reencoded_all[:-12]
y_for_lda = Y_valid

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_for_lda, y_for_lda)

X_lda_transformed_extra = lda.transform(X_reencoded_all[-12:])
X_lda_translated = X_lda_transformed_extra[:-2]
X_lda_src_centroid = X_lda_transformed_extra[-2]
X_lda_dst_centroid = X_lda_transformed_extra[-1]

plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_lda[:, 0],
    X_lda[:, 1],
    c=y_for_lda,
    cmap="Paired",
    alpha=0.35,
    s=6
)

unique_classes = np.unique(Y_valid)
norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

plt.scatter(X_lda_src_centroid[0], X_lda_src_centroid[1], marker="x", color="red", s=100, label="Centroïde source")
plt.scatter(X_lda_dst_centroid[0], X_lda_dst_centroid[1], marker="x", color="blue", s=100, label="Centroïde destination")

plt.arrow(X_lda_src_centroid[0], X_lda_src_centroid[1],
          X_lda_dst_centroid[0] - X_lda_src_centroid[0],
          X_lda_dst_centroid[1] - X_lda_src_centroid[1],
          color="black", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (centres)")

for i in range(10):
    src = X_lda[digits[src_class][i]]
    dst = X_lda_translated[i]
    plt.scatter(src[0], src[1], marker="+", color="red", s=150)
    plt.scatter(dst[0], dst[1], marker="+", color="blue", s=150)

    plt.arrow(src[0], src[1],
              dst[0] - src[0],
              dst[1] - src[1],
              color="purple", width=0.01, head_width=0.2, length_includes_head=True)
    
plt.scatter([], [], marker="+", color="red", label="Chiffre source", s=150)
plt.scatter([], [], marker="+", color="blue", label="Chiffre translaté", s=150)
plt.arrow([], [], [], [], color="purple", width=0.01, head_width=0.2, length_includes_head=True, label="Translation (chiffres)")

plt.title(f"LDA : Translation de {src_class} vers {dst_class}")
plt.legend()
plt.tight_layout()
plt.savefig("./Results/mnist-translation-lda.png")