# c'est le code le plus triste que j'ai fait pendant ce stage

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

import seaborn as sns

from keras.datasets import mnist

import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical
from sklearn.manifold import TSNE

import itertools
import cv2

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

idx1 = np.concatenate([np.where(Y_train == c)[0][:len(np.where(Y_train == c)[0]) // 2] for c in range(10)])
idx2 = np.concatenate([np.where(Y_train == c)[0][len(np.where(Y_train == c)[0]) // 2:] for c in range(10)])

idx1 = tf.convert_to_tensor(idx1, dtype=tf.int32)
idx2 = tf.convert_to_tensor(idx2, dtype=tf.int32)

X_split1 = tf.gather(X_train, idx1)
Y_split1 = tf.gather(Y_train, idx1)

X_split2 = tf.gather(X_train, idx2)
Y_split2 = tf.gather(Y_train, idx2)

X_test_full = np.concatenate((X_split2, X_valid))
Y_test_full = np.concatenate((Y_split2, Y_valid))

X_classes = [X_test_full[Y_test_full == i] for i in range(10)]
Y_classes = [Y_test_full[Y_test_full == i] for i in range(10)]

tc = 0.75

split_index = [int(tc * len(cls)) for cls in X_classes]

X_classes2 = [cls[idx:] for cls, idx in zip(X_classes, split_index)]
X_classes_original2 = np.array(list(itertools.chain(*X_classes2)))

Y_classes2 = [cls[idx:] for cls, idx in zip(Y_classes, split_index)]
Y_classes_original2 = np.array(list(itertools.chain(*Y_classes2)))

encoder = load_model("./Models/DISVAE/mnist-128-h-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-h-decoder.keras")

encoded_means = utils.encoded_means(X_split1, Y_split1, "h_encoded_means_disvae", encoder, decoder, 2, 32)
encoded_std = utils.encoded_std(X_split1, Y_split1, "h_encoded_std_disvae", encoder, decoder, 2, 32)

X_classes_inverse2 = np.empty((0, 64, 64, 1))
Y_classes_translated2 = np.array([])
Y_classes_isTranslated2 = np.array([])

for src_class in range(10):
    src_classes = np.array_split(X_classes2[src_class], 10)

    for dst_class in range(10):
        print(src_class, dst_class)
        Y_classes_translated2 = np.append(Y_classes_translated2, np.full(len(src_classes[dst_class]), dst_class))
        Y_classes_isTranslated2 = np.append(Y_classes_isTranslated2, np.full(len(src_classes[dst_class]), int(src_class != dst_class)))

        if src_class == dst_class:
            X_classes_inverse2 = np.concatenate((X_classes_inverse2, src_classes[dst_class]), axis=0)
            continue

        X_encoded_src = utils.encoded(src_classes[dst_class], "", encoder, decoder, 3, 32, False)
        X_translated = encoded_means[dst_class] + (encoded_std[dst_class] / encoded_std[src_class]) * (X_encoded_src - encoded_means[src_class])
        src_classes[dst_class] = decoder.predict(X_translated, batch_size = 32)

        X_reencoded_src = encoder.predict(src_classes[dst_class], batch_size = 32)
        X_inverse_translated = encoded_means[src_class] + (encoded_std[src_class] / encoded_std[dst_class]) * (X_reencoded_src - encoded_means[dst_class])
        X_redecoded = decoder.predict(X_inverse_translated, batch_size = 32)

        X_classes_inverse2 = np.concatenate((X_classes_inverse2, X_redecoded), axis=0)

    X_classes2[src_class] = np.concatenate(src_classes)

Y_classes2 = np.repeat(np.arange(10), np.array([len(src_class) for src_class in X_classes2]))
Y_classes_cat2 = to_categorical(Y_classes2, 10)

X_classes2 = np.array(list(itertools.chain(*X_classes2)))
X_classes_original2 = tf.image.resize(X_classes_original2, (28, 28))
X_classes2 = tf.image.resize(X_classes2, (28, 28))
X_classes_inverse2 = tf.image.resize(X_classes_inverse2, (28, 28))

indices = np.arange(X_classes2.shape[0])
np.random.shuffle(indices)
indices = tf.convert_to_tensor(indices, dtype=tf.int32)
X_classes2 = tf.gather(X_classes2, indices)
X_classes_inverse2 = tf.gather(X_classes_inverse2, indices)
Y_classes2 = tf.gather(Y_classes2, indices)
Y_classes_cat2 = tf.gather(Y_classes_cat2, indices)
Y_classes_translated2 = tf.gather(Y_classes_translated2, indices)
Y_classes_isTranslated2 = tf.gather(Y_classes_isTranslated2, indices)

classifier = load_model("./Models/Classifieur/classifier.keras")
res_classifier = load_model("./Models/Classifieur/residual-classifier-std-128.keras")
detect_classifier = load_model("./Models/Classifieur/residual-detection-classifier-128.keras")

# Reconnaissance de la classe sans translation (détection de trace)
Y_pred = res_classifier.predict(X_classes_original2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes_original2, Y_pred_classes)

cm = confusion_matrix(Y_classes_original2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Reconnaissances des classes sans translation par détection de trace", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-complete-unchanged-classifier-confusion.png")

Y_pred_guessed_src_classes = Y_pred_classes

# Reconnaissance de la classe source (détection de trace)
Y_pred = res_classifier.predict(X_classes2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes2, Y_pred_classes)

cm = confusion_matrix(Y_classes2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des classes sources", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion.png")

Y_pred_guessed_dst_classes = Y_pred_classes

# Reconnaissance de la classe destination par translation inverse (détection de trace)
Y_pred = res_classifier.predict(X_classes_inverse2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes_translated2, Y_pred_classes)

cm = confusion_matrix(Y_classes_translated2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des classes destination après translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion-inverse-dst.png")

# Reconnaissance de la classe source par translation inverse (détection de trace)
Y_pred = res_classifier.predict(X_classes_inverse2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes2, Y_pred_classes)

cm = confusion_matrix(Y_classes2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des classes sources après translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion-inverse-src.png")

# Reconnaissance de la classe prédite par translation inverse (détection de trace)
Y_pred = res_classifier.predict(X_classes_inverse2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_pred_guessed_dst_classes, Y_pred_classes)

cm = confusion_matrix(Y_pred_guessed_dst_classes, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des classes destination prédites, après translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion-inverse-predicted-dst.png")

# Reconnaissance de la classe source prédite par translation inverse (détection de trace)
Y_pred = res_classifier.predict(X_classes_inverse2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_pred_guessed_src_classes, Y_pred_classes)

cm = confusion_matrix(Y_pred_guessed_src_classes, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des classes sources prédites, après translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion-inverse-predicted-src.png")

# Reconnaissance de la classe destination (classification naïve)
Y_pred = classifier.predict(X_classes2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes_translated2, Y_pred_classes)

cm = confusion_matrix(Y_classes_translated2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Classification", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-normal-classifier-confusion.png")

# Reconnaissance de la classe d'origine après translation inverse (classification naïve)
Y_pred = classifier.predict(X_classes_inverse2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_classes2, Y_pred_classes)

cm = confusion_matrix(Y_classes2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Classification après translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-normal-classifier-confusion-inverse.png")

# Détection de la translation
Y_pred = detect_classifier.predict(X_classes2)

Y_pred_classes = (Y_pred >= 0.5).astype(int)

accuracy = accuracy_score(Y_classes_isTranslated2, Y_pred_classes)

average_certainty = 1.0 - np.mean(np.abs(Y_pred - Y_pred_classes))

cm = confusion_matrix(Y_classes_isTranslated2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

detection_labels = ["Non détecté", "Détecté"]

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=detection_labels, yticklabels=detection_labels, vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection de la translation", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-detection-translation-confusion.png")

# Détection de la translation inverse
Y_pred = detect_classifier.predict(X_classes_inverse2)

Y_pred_classes = (Y_pred >= 0.5).astype(int)

accuracy = accuracy_score(Y_classes_isTranslated2, Y_pred_classes)

average_certainty = 1.0 - np.mean(np.abs(Y_pred - Y_pred_classes))

cm = confusion_matrix(Y_classes_isTranslated2, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

detection_labels = ["Non détecté", "Détecté"]

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=detection_labels, yticklabels=detection_labels, vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection de la translation inverse", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-detection-translation-confusion-inverse.png")

# Translation d'une image réelle
image_path = "./Images/2.jpg"
image = cv2.imread(image_path)

image64 = cv2.resize(image, (64, 64))
image = cv2.cvtColor(image64, cv2.COLOR_BGR2GRAY)
threshold_value = 128
_, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
image = cv2.bitwise_not(image)

image = image.astype("float32") / 255.
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

fig, axes = plt.subplots(2, 13, figsize=(20, 5))
plt.subplots_adjust(hspace=0.5)

axes[0, 0].imshow(cv2.cvtColor(image64, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Image originale")
axes[0, 0].axis("off")

predicted = utils.encoded(image, "", encoder, decoder, 3, 1, False)

axes[0, 1].imshow(image[0], cmap="gray")
axes[0, 1].set_title("Image seuillée")
axes[0, 1].axis("off")

src_class, p, linp = utils.classify(image, classifier)
src_class_g, p_g, linp_g = utils.classify(image, res_classifier)
is_translated, p_d, certainty = utils.classify_binary(image, detect_classifier)
axes[0, 1].text(0.5, -0.15, f"({src_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 1].transAxes)
axes[0, 1].text(0.5, -0.3, f"({src_class_g}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[0, 1].transAxes)
axes[0, 1].text(0.5, -0.45, f"({is_translated.squeeze()}, {certainty.squeeze():.3f})", fontsize=14, color="green", ha="center", transform=axes[0, 1].transAxes)

decoded = decoder.predict(predicted)

axes[0, 2].imshow(decoded[0], cmap="gray")
axes[0, 2].set_title("Reconstruction")
axes[0, 2].axis("off")

guessed_class, p, linp = utils.classify(decoded, classifier)
guessed_class_g, p_g, linp_g = utils.classify(decoded, res_classifier)
is_translated, p_d, certainty = utils.classify_binary(decoded, detect_classifier)
axes[0, 2].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, -0.3, f"({guessed_class_g}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, -0.45, f"({is_translated.squeeze()}, {certainty.squeeze():.3f})", fontsize=14, color="green", ha="center", transform=axes[0, 2].transAxes)

axes[1, 0].axis("off")
axes[1, 1].axis("off")
axes[1, 2].axis("off")

translated_encoded = []
translated_decoded_images = []
for dst_class in range(10):
    translated = predicted + encoded_means[dst_class] - encoded_means[src_class]
    if dst_class != src_class:
        translated_encoded.append(translated)
    translated_decoded = decoder.predict(translated)
    translated_decoded_images.append(translated_decoded)

    axes[0, dst_class + 3].imshow(translated_decoded[0], cmap="gray")
    axes[0, dst_class + 3].set_title(f"{dst_class}")
    axes[0, dst_class + 3].axis("off")

    guessed_class, p, linp = utils.classify(translated_decoded, classifier)
    guessed_class_g, p_g, linp_g = utils.classify(translated_decoded, res_classifier)
    is_translated, p_d, certainty = utils.classify_binary(translated_decoded, detect_classifier)

    axes[0, dst_class + 3].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[0, dst_class + 3].transAxes)
    axes[0, dst_class + 3].text(0.5, -0.3, f"({guessed_class_g}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[0, dst_class + 3].transAxes)
    axes[0, dst_class + 3].text(0.5, -0.45, f"({is_translated.squeeze()}, {certainty.squeeze():.3f})", fontsize=14, color="green", ha="center", transform=axes[0, dst_class + 3].transAxes)

    translated_reencoded = encoder.predict(translated_decoded)
    inverse_translated = translated_reencoded + encoded_means[src_class] - encoded_means[dst_class]
    inverse_decoded = decoder.predict(inverse_translated)
    
    axes[1, dst_class + 3].imshow(inverse_decoded[0], cmap="gray")
    axes[1, dst_class + 3].axis("off")

    guessed_class, p, linp = utils.classify(inverse_decoded, classifier)
    guessed_class_g, p_g, linp_g = utils.classify(inverse_decoded, res_classifier)
    is_translated, p_d, certainty = utils.classify_binary(inverse_decoded, detect_classifier)

    axes[1, dst_class + 3].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[1, dst_class + 3].transAxes)
    axes[1, dst_class + 3].text(0.5, -0.3, f"({guessed_class_g}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[1, dst_class + 3].transAxes)
    axes[1, dst_class + 3].text(0.5, -0.45, f"({is_translated.squeeze()}, {certainty.squeeze():.3f})", fontsize=14, color="green", ha="center", transform=axes[1, dst_class + 3].transAxes)

translated_encoded = np.array(translated_encoded).squeeze()

plt.tight_layout()
plt.savefig("./Results/mnist-trace-translated-image.png")

# t-SNE
X_classes2 = tf.image.resize(X_classes2, (64, 64))
X_encoded_classes = encoder.predict(X_classes2)
X_encoded = np.concatenate((X_encoded_classes, predicted, translated_encoded, encoded_means.squeeze()))

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)
X_tsne_full = tsne.fit_transform(X_encoded)
X_tsne = X_tsne_full[:-20]
X_predicted = X_tsne_full[-20]
X_translated = X_tsne_full[-19:-10]
X_encoded_means = X_tsne_full[-10:]

# t-SNE destination + Translation de l'image réelle
plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=Y_classes2,
    cmap="Paired",
    alpha=0.35,
    s=20
)

unique_classes = np.unique(Y_valid)
norm = Normalize(vmin = min(unique_classes), vmax = max(unique_classes))

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

std = 1.5
for class_label in unique_classes:
    mask = (Y_classes_translated2 == class_label)
    class_data = X_tsne[mask]

    mean = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    width, height = 2 * std * np.sqrt(eigvals)

    color = plt.cm.Paired(norm(class_label))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=color, edgecolor=color, alpha=0.35, lw=3, zorder=0)
    plt.gca().add_patch(ellipse)

plt.scatter(
    X_encoded_means[:, 0],
    X_encoded_means[:, 1],
    color="black",
    marker='x',
    s=100,
    linewidths=2,
    label="Centroïde"
)

plt.title(f"t-SNE : Translations de plusieurs sources vers les clusters des classes")
plt.legend()
plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-res-full-tsne-translations.png")

# Origine des images translatées et inchangées sur le t-SNE
plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=Y_classes_translated2,
    cmap="Paired",
    alpha=0.35,
    s=20
)

unique_classes = np.unique(Y_valid)
norm = Normalize(vmin = min(unique_classes), vmax = max(unique_classes))

for class_label in unique_classes:
    plt.scatter([], [], color=plt.cm.Paired(norm(class_label)), label=str(class_label))

for i in range(9):
    plt.scatter(X_predicted[0], X_predicted[1], marker="+", color="red", s=150)
    plt.scatter(X_translated[i][0], X_translated[i][1], marker="+", color="blue", s=150)

    plt.arrow(X_predicted[0], X_predicted[1],
            X_translated[i][0] - X_predicted[0],
            X_translated[i][1] - X_predicted[1],
            color="purple", width=0.01, head_width=0.2, length_includes_head=True)

plt.scatter([], [], marker="+", color="red", s=150, label="Image source")
plt.scatter([], [], marker="+", color="blue", s=150, label="Image translatée")

plt.scatter(
    X_encoded_means[:, 0],
    X_encoded_means[:, 1],
    color="black",
    marker='x',
    s=100,
    linewidths=2,
    label="Centroïde"
)

plt.title(f"t-SNE : Translations de l'image réelle vers les clusters des classes")
plt.legend()
plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-res-full-tsne.png")