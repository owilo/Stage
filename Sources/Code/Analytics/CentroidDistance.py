import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist

from Code.Models import BetaVAE
from Code.Utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_class_distributions = latent.class_distributions_n(
    autoencoder,
    x=x_train,
    y=y_train,
    n=2,
    save_cache=True
)

z_test = latent.encode_n(
    autoencoder,
    x=x_test,
    y=y_test,
    n=3,
    save_cache=True
)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_distance(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return 1 - dot / (norm1 * norm2)

eucl_dist_matrix = np.zeros((10, 10), dtype=float)
cos_dist_matrix = np.zeros((10, 10), dtype=float)

for src_class in range(10):
    digits = z_test[y_test == src_class]

    for cnt_class in range(10):
        mean, _ = z_class_distributions[cnt_class]

        eucl_distances = []
        cos_distances = []
        
        for digit in digits:
            eucl_distances.append(euclidean_distance(digit, mean))
            cos_distances.append(cosine_distance(digit, mean.flatten()))
        
        eucl_distances = np.array(eucl_distances)
        cos_distances = np.array(cos_distances)
        
        avg_eucl_distance = np.mean(eucl_distances, axis=0)
        avg_cos_distance = np.mean(cos_distances, axis=0)
        
        eucl_dist_matrix[src_class, cnt_class] = avg_eucl_distance
        cos_dist_matrix[src_class, cnt_class] = avg_cos_distance

plt.figure(figsize=(10, 8))
sns.heatmap(eucl_dist_matrix, annot=True, cmap="Reds", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
plt.title("Distance euclidienne moyenne des chiffres aux centroïdes des classes", fontsize=14)
plt.xlabel("Centroïde")
plt.ylabel("Classe source")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Distances" / "mnist-eucl-distance-centroid.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cos_dist_matrix, annot=True, cmap="Reds", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
plt.title("Distance cosinus moyenne des chiffres aux centroïdes des classes", fontsize=14)
plt.xlabel("Centroïde")
plt.ylabel("Classe source")
plt.tight_layout()
plt.savefig(cache.RESULTS_FOLDER / "Distances" / "mnist-cosine-distance-centroid.png")
plt.close()

for src_class in range(10):
    eucl_dist_matrix = np.zeros((10, 10), dtype=float)
    cos_dist_matrix = np.zeros((10, 10), dtype=float)

    z_test_class = z_test[y_test == src_class]
    
    y_src = np.full(len(z_test_class), src_class)
    
    for dst_class in range(10):
        y_dst = np.full(len(z_test_class), dst_class)
        z_dst = latent.translate(z_test_class, y_src, y_dst, z_class_distributions)
        
        for cnt_class in range(10):
            mean, _ = z_class_distributions[cnt_class]

            eucl_distances = []
            cos_distances = []
            
            for z in z_dst:
                eucl_distances.append(euclidean_distance(z, mean))
                cos_distances.append(cosine_distance(z, mean.flatten()))
            
            eucl_distances = np.array(eucl_distances)
            cos_distances = np.array(cos_distances)
            
            avg_eucl_distance = np.mean(eucl_distances, axis=0)
            avg_cos_distance = np.mean(cos_distances, axis=0)
            
            eucl_dist_matrix[dst_class, cnt_class] = avg_eucl_distance
            cos_dist_matrix[dst_class, cnt_class] = avg_cos_distance

    plt.figure(figsize=(10, 8))
    sns.heatmap(eucl_dist_matrix, annot=True, cmap="Reds", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
    plt.suptitle(f"Classe source {src_class}", fontsize=22)
    plt.title("Distance euclidienne moyenne des chiffres translatés aux centroïdes des classes", fontsize=14)
    plt.xlabel("Centroïde")
    plt.ylabel("Classe translatée")
    plt.tight_layout()
    plt.savefig(cache.RESULTS_FOLDER / "Distances" / f"mnist-eucl-distance-centroid-translated-{src_class}.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_dist_matrix, annot=True, cmap="Reds", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
    plt.suptitle(f"Classe source {src_class}", fontsize=22)
    plt.title("Distance cosinus moyenne des chiffres translatés aux centroïdes des classes", fontsize=14)
    plt.xlabel("Centroïde")
    plt.ylabel("Classe translatée")
    plt.tight_layout()
    plt.savefig(cache.RESULTS_FOLDER / "Distances" / f"mnist-cosine-distance-centroid-translated-{src_class}.png")
    plt.close()