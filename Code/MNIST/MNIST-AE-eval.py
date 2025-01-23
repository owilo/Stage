import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import math

from sklearn.manifold import TSNE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow.keras.backend as K # backend différent de keras.backend
from tensorflow.keras.models import load_model

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_eval = np.concatenate((X_train, X_valid))
Y_eval = np.concatenate((Y_train, Y_valid))
# todo : remplacer les X_valid et Y_valid par eval... !!!NOTE : les digits changent

num_epochs = 6

img_shape = (28, 28, 1)
batch_size = 16

latent_dims = [8, 32, 64, 128]

ae_type = "VAE"

X_eval_encoded = [None] * len(latent_dims)
X_decoded = [None] * len(latent_dims)

decoder = [None] * len(latent_dims)

for li in range(len(latent_dims)):
    latent_dim = latent_dims[li]

    encoder = load_model("./Models/" + ae_type + "/mnist-" + str(latent_dim) + "-encoder.keras")
    decoder[li] = load_model("./Models/" + ae_type + "/mnist-" + str(latent_dim) + "-decoder.keras")

    print("Encoding (" + str(latent_dim) + ")")
    X_eval_encoded[li] = encoder.predict(X_eval, batch_size = batch_size)

    print("Decoding (" + str(latent_dim) + ")")
    X_decoded[li] = decoder[li].predict(X_eval_encoded[li], batch_size = batch_size)

# Évaluation des reconstructions de digits

print("Collecting results...")

"""digits = [
    [1333, 5526, 972, 5554, 5838],
    [9415, 5889, 672, 831, 7344],
    [3773, 1256, 2082, 3513, 3047],
    [524, 5358, 313, 590, 3085],
    [1980, 4217, 5713, 5200, 3985],
    [1874, 6491, 7583, 3220, 5473],
    [4252, 446, 4466, 7668, 3135],
    [6960, 75, 3309, 9141, 6050],
    [8466, 5729, 7745, 128, 4380],
    [5333, 1088, 1869, 4683, 1005]
]"""
# TODO note : +60000
digits = [
    [61333, 65526, 60972, 65554, 65838],
    [69415, 65889, 60672, 60831, 67344],
    [63773, 61256, 62082, 63513, 63047],
    [60524, 65358, 60313, 60590, 63085],
    [61980, 64217, 65713, 65200, 63985],
    [61874, 66491, 67583, 63220, 65473],
    [64252, 60446, 64466, 67668, 63135],
    [66960, 60075, 63309, 69141, 66050],
    [68466, 65729, 67745, 60128, 64380],
    [65333, 61088, 61869, 64683, 61005]
]


for i in range(10):
    fig, axes = plt.subplots(5, 1 + len(latent_dims), figsize=(2 * (1 + len(latent_dims)), 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    fig.suptitle("Dimension de l'espace latent", x=0.5, y=0.92, fontsize=18)
    axes[0, 0].set_title("Source", fontsize=14)

    for j in range(len(digits[i])):
        digit_index = digits[i][j]

        axes[j, 0].imshow(X_eval[digit_index].reshape(28, 28))
        axes[j, 0].axis("off")

    for j in range(len(latent_dims)):
        axes[0, 1 + j].set_title(str(latent_dims[j]), fontsize=14)

        for k in range(len(digits[i])):
            digit_index = digits[i][k]
            
            img_src = X_eval[digit_index].reshape(28, 28)
            img_dst = X_decoded[j][digit_index].reshape(28, 28)

            axes[k, 1 + j].imshow(img_dst)

            decoded_psnr = psnr(img_src, img_dst)
            decoded_ssim = ssim(img_src, img_dst, data_range = 1)

            axes[k, 1 + j].axis("off")
            axes[k, 1 + j].text(0.5, -0.4, f"PSNR = {decoded_psnr:.2f} dB\nSSIM = {decoded_ssim:.2f}", fontsize = 12, color = "blue", ha = "center", transform = axes[k, 1 + j].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-" + str(i) + ".png")

# Évaluation des reconstructions moyennes de digits

print("Collecting average results...")

avgs_n = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.002]

for i in range(10):
    fig, axes = plt.subplots(len(avgs_n), len(latent_dims), figsize=(2 * len(latent_dims), 2 * len(avgs_n)))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    fig.suptitle("Dimension de l'espace latent", x=0.5, y=0.92, fontsize=18)
    for j in range(len(avgs_n)):
        prcnt = avgs_n[j] * 100.0
        axes[j, 0].annotate(f"{prcnt:.2f}%", xy=(-0.2, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=12)

    for j in range(len(latent_dims)):
        axes[0, j].set_title(str(latent_dims[j]), fontsize=14)

        for k in range(len(avgs_n)):
            X_digit_encoded = X_eval_encoded[j][Y_eval == i]

            indices = np.random.choice(X_digit_encoded.shape[0], math.ceil(avgs_n[k] * X_digit_encoded.shape[0]), replace=False)

            random_encoded = X_digit_encoded[indices]
        
            mean_encoded = np.mean(random_encoded, axis = 0).reshape(1, -1)

            X_decoded_avg = decoder[j].predict(mean_encoded, batch_size = batch_size, verbose = False)
            axes[k, j].imshow(X_decoded_avg[0])
            axes[k, j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-avg-" + str(i) + ".png")

# Calcul du PSNR et SSIM moyens

print("Calculating mean metrics...")

output = "latent_size,psnr,ssim\n"
for i in range(len(latent_dims)):
    mean_psnr = np.mean([psnr(X_eval[j], X_decoded[i][j]) for j in range(X_eval.shape[0])])
    mean_ssim = np.mean([ssim(X_eval[j], X_decoded[i][j], data_range = 1, channel_axis = -1) for j in range(X_eval.shape[0])])
    output += str(latent_dims[i]) + "," + str(mean_psnr) + "," + str(mean_ssim) + "\n"

with open("./Results/Model-Eval/" + ae_type + "/mnist-eval-metrics.csv", 'w') as f:
    f.write(output)

# t-SNE

plt.close("all")

print("Calculating t-SNE projection...")

tsne = TSNE(n_components = 2, random_state = 42, max_iter = 300)
for i in range(len(latent_dims)):
    X_eval_encoded_tSNE = tsne.fit_transform(X_eval_encoded[i])

    plt.figure(figsize=(8, 8))

    colors = colormaps["Paired"].colors
    for j in range(10):
        indices = (Y_eval == j)
        plt.scatter(
            X_eval_encoded_tSNE[indices, 0],
            X_eval_encoded_tSNE[indices, 1],
            label = str(j),
            alpha = 0.25,
            color = colors[j]
        )

        centroid = X_eval_encoded_tSNE[indices].mean(axis = 0)
        
        plt.scatter(centroid[0], centroid[1], marker = "x", color = "black", s = 100)

    plt.legend()
    plt.title("latent t-SNE")
    plt.tight_layout()
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-tsne-" + str(latent_dims[i]) + ".png")

"""
tsne = TSNE(n_components = 2, random_state = 42, max_iter = 300)
for i in range(len(latent_dims)):
    X_eval_encoded_tSNE = tsne.fit_transform(X_eval_encoded[i])

    plt.figure(figsize=(8, 8))

    colors = colormaps["Paired"].colors

    for i in range(X_eval_encoded_tSNE.shape[0]):
        label = Y_eval[i]
        plt.scatter(
            X_eval_encoded_tSNE[i, 0],
            X_eval_encoded_tSNE[i, 1],
            label = str(label),
            color = colors[label],
        )

    # Calculate and plot centroids
    for j in range(10):
        indices = (Y_eval == j)
        centroid = X_eval_encoded_tSNE[indices].mean(axis=0)
        plt.scatter(centroid[0], centroid[1], marker = "x", color = "black", s = 100)

    plt.legend()
    plt.title("Latent t-SNE")
    plt.tight_layout()
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-tsne-" + str(latent_dims[i]) + ".png")
"""