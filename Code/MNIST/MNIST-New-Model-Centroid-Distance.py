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

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/VAE/mnist-128-encoder-dis2.keras")
decoder = load_model("./Models/VAE/mnist-128-decoder-dis2.keras")

X_rereencoded_valid = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_distance(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return 1 - dot / (norm1 * norm2)

for src_class in range(10):
    eucl_dist_matrix = np.zeros((10, 10), dtype=float)
    cos_dist_matrix = np.zeros((10, 10), dtype=float)

    digits = X_rereencoded_valid[Y_valid == src_class]
    mean_encoded_src = encoded_means[src_class]
    
    for dst_class in range(10):
        mean_encoded_dst = encoded_means[dst_class]
        translation = mean_encoded_dst - mean_encoded_src
        translated = digits + translation
        
        for cnt_class in range(10):
            eucl_distances = []
            cos_distances = []
            
            for translated_digit in translated:
                eucl_distances.append(euclidean_distance(translated_digit, encoded_means[cnt_class]))
                cos_distances.append(cosine_distance(translated_digit, encoded_means[cnt_class].flatten()))
            
            eucl_distances = np.array(eucl_distances)
            cos_distances = np.array(cos_distances)
            
            avg_eucl_distance = np.mean(eucl_distances, axis=0)
            avg_cos_distance = np.mean(cos_distances, axis=0)
            
            eucl_dist_matrix[dst_class, cnt_class] = avg_eucl_distance
            cos_dist_matrix[dst_class, cnt_class] = avg_cos_distance

    plt.figure(figsize=(10, 8))
    sns.heatmap(eucl_dist_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
    plt.suptitle(f"Classe source {src_class}", fontsize=22)
    plt.title("Distance euclidienne moyenne des chiffres translatés aux centroïdes des classes", fontsize=14)
    plt.xlabel("Centroïde")
    plt.ylabel("Classe translatée")
    plt.savefig(f"./Results/Distances/mnist-eucl-distance-centroid-{src_class}.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_dist_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
    plt.suptitle(f"Classe source {src_class}", fontsize=22)
    plt.title("Distance cosinus moyenne des chiffres translatés aux centroïdes des classes", fontsize=14)
    plt.xlabel("Centroïde")
    plt.ylabel("Classe translatée")
    plt.savefig(f"./Results/Distances/mnist-cosine-distance-centroid-{src_class}.png")
    plt.close()