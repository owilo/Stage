import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import math
import sys
import os

from sklearn.manifold import TSNE
from sklearn.metrics import precision_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from keras.datasets import mnist

import tensorflow.keras.backend as K # backend différent de keras.backend
from tensorflow.keras.models import load_model

import os
import numpy as np

def cache_array(filename, array_generator, save_cache=True, verbose=True):
    file_path = os.path.join("./Cache", filename)

    if os.path.exists(file_path):
        if (verbose):
            print(f"Chargement des données depuis {filename}")
        return np.load(file_path)
    else:
        array = array_generator()
        if (save_cache):
            if (verbose):
                print(f"Sauvegarde des données dans {filename}")
            np.save(file_path, array)
        return array

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_eval = np.concatenate((X_train, X_valid))
Y_eval = np.concatenate((Y_train, Y_valid))

img_shape = (28, 28, 1)
batch_size = 16

latent_dims = [8, 32, 64, 128]

if len(sys.argv) < 2:
    print("Format :\n\t - Type d'auto-encodeur : 'AE' (simple) ou 'VAE'")
    exit(1)

ae_type = "AE" if sys.argv[1] == "AE" else "VAE"

X_eval_encoded = [None] * len(latent_dims)
X_decoded = [None] * len(latent_dims)

decoder = [None] * len(latent_dims)

for li in range(len(latent_dims)):
    latent_dim = latent_dims[li]

    encoder = load_model("./Models/" + ae_type + "/mnist-" + str(latent_dim) + "-encoder.keras")
    decoder[li] = load_model("./Models/" + ae_type + "/mnist-" + str(latent_dim) + "-decoder.keras")

    print("Encoding (" + str(latent_dim) + ")")
    X_eval_encoded[li] = cache_array(f"{ae_type}-encoded-{latent_dim}.npy", lambda: encoder.predict(X_eval, batch_size = batch_size))

    print("Decoding (" + str(latent_dim) + ")")
    X_decoded[li] = cache_array(f"{ae_type}-decoded-{latent_dim}.npy", lambda: decoder[li].predict(X_eval_encoded[li], batch_size = batch_size))

classifier = load_model("./Models/Classifieur/classifier.keras")

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
    [61333, 65526, 60972, 65554, 65838], # 0
    [69415, 65889, 60672, 60831, 67344], # 1
    [63773, 61256, 62082, 63513, 63047], # 2
    [60524, 65358, 60313, 60590, 63085], # 3
    [61980, 64217, 65713, 65200, 63985], # 4
    [61874, 66491, 67583, 63220, 65473], # 5
    [64252, 60446, 64466, 67668, 63135], # 6
    [66960, 60075, 63309, 69141, 66050], # 7
    [68466, 65729, 67745, 60128, 64380], # 8
    [65333, 61088, 61869, 64683, 61005]  # 9
]


for i in range(10):
    fig, axes = plt.subplots(5, 1 + len(latent_dims), figsize=(2 * (1 + len(latent_dims)), 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

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
            decoded_ssim = ssim(img_src, img_dst, data_range=1)

            img_dst_reshaped = img_dst.reshape(1, 28, 28, 1)

            Y_pred_proba = classifier.predict(img_dst_reshaped, verbose = False)

            guessed_class = np.argmax(Y_pred_proba)
            certainty = np.max(Y_pred_proba)

            axes[k, 1 + j].axis("off")
            axes[k, 1 + j].text(0.5, -0.7, f"PSNR = {decoded_psnr:.2f} dB\nSSIM = {decoded_ssim:.2f}\n({guessed_class}, {certainty:.3f})", fontsize=12, color="blue", ha="center", transform=axes[k, 1 + j].transAxes)


    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-" + str(i) + ".png")

# Évaluation des reconstructions moyennes de digits

print("Collecting average results...")

avgs_n = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0002]

correct_guesses = [0] * len(latent_dims)
cumulated_certainty = [0] * len(latent_dims)

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

            nb_idx = math.ceil(avgs_n[k] * X_digit_encoded.shape[0])

            indices = np.random.choice(X_digit_encoded.shape[0], nb_idx, replace=False)

            random_encoded = X_digit_encoded[indices]
        
            mean_encoded = np.mean(random_encoded, axis = 0).reshape(1, -1)

            X_decoded_avg = decoder[j].predict(mean_encoded, batch_size = batch_size, verbose = False)[0]

            X_decoded_avg_reshaped = X_decoded_avg.reshape(1, 28, 28, 1)

            Y_pred_proba = classifier.predict(X_decoded_avg_reshaped, verbose = False)

            guessed_class = np.argmax(Y_pred_proba)
            certainty = np.max(Y_pred_proba)

            if guessed_class == i:
                correct_guesses[j] += 1
            cumulated_certainty[j] += certainty

            axes[k, j].imshow(X_decoded_avg)
            axes[k, j].axis("off")
            axes[k, j].text(0.5, -0.15, f"({guessed_class}, {certainty:.3f})", fontsize=12, color="blue", ha="center", transform=axes[k, j].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("./Results/Model-Eval/" + ae_type + "/mnist-eval-avg-" + str(i) + ".png")

output = ",".join(map(str, latent_dims)) + "\n"

precision = [correct_guesses[i] / (10.0 * len(avgs_n)) for i in range(len(latent_dims))]
certainty = [cumulated_certainty[i] / (10.0 * len(avgs_n)) for i in range(len(latent_dims))]

output += ",".join(map(str, precision)) + "\n"
output += ",".join(map(str, certainty)) + "\n"
with open("./Results/Model-Eval/" + ae_type + "/mnist-eval-avg-classification.csv", 'w') as f:
    f.write(output)

# Calcul du PSNR et SSIM moyens

print("Calculating mean metrics...")

output = "latent_size,psnr,ssim,class_prec,class_certainty\n"

for i in range(len(latent_dims)):
    mean_psnr = np.mean([psnr(X_eval[j], X_decoded[i][j]) for j in range(X_eval.shape[0])])
    mean_ssim = np.mean([ssim(X_eval[j], X_decoded[i][j], data_range = 1, channel_axis = -1) for j in range(X_eval.shape[0])])

    Y_pred_proba = classifier.predict(X_decoded[i])
    Y_pred = np.argmax(Y_pred_proba, axis=1)

    certainty = np.max(Y_pred_proba, axis=1)
    average_certainty = np.mean(certainty)

    precision = precision_score(Y_eval, Y_pred, average="macro")

    output += str(latent_dims[i]) + "," + str(mean_psnr) + "," + str(mean_ssim) + "," + str(precision) + "," + str(average_certainty) + "\n"

with open("./Results/Model-Eval/" + ae_type + "/mnist-eval-metrics.csv", 'w') as f:
    f.write(output)

output = "classe," + ",".join(map(str, latent_dims)) + "\n"

for i in range(10):
    stds = [0.0] * len(latent_dims)
    for j in range(len(latent_dims)):
        stds[j] = np.std(X_eval_encoded[j][Y_eval == i], axis = 0).mean()
    output += str(i) + "," + ",".join(map(str, stds)) + "\n"

with open("./Results/Model-Eval/" + ae_type + "/mnist-eval-std.csv", 'w') as f:
    f.write(output)

# t-SNE

plt.close("all")

print("Calculating t-SNE projection...")

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)
for i in range(len(latent_dims)):
    X_eval_encoded_tSNE = cache_array(f"{ae_type}-encoded-tsne-{latent_dims[i]}.npy", lambda: tsne.fit_transform(X_eval_encoded[i]))

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        X_eval_encoded_tSNE[:, 0],
        X_eval_encoded_tSNE[:, 1],
        c=Y_eval,
        cmap="Paired",
        alpha=0.35,
        s=6
    )

    for j in range(10):
        indices = (Y_eval == j)
        centroid = X_eval_encoded_tSNE[indices].mean(axis=0)
        plt.scatter(centroid[0], centroid[1], marker="x", color="black", s=100)

    unique_classes = np.unique(Y_eval)
    norm = Normalize(vmin = min(unique_classes), vmax = max(unique_classes))

    for class_label in unique_classes:
        plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

    plt.title("latent t-SNE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Results/Model-Eval/{ae_type}/mnist-eval-tsne-{latent_dims[i]}.png")
