import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import seaborn as sns

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

import cv2

import Utils

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

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

encoded_means = Utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

src_class0 = 2
src_class1 = 7
dst_class = 5

tc = 0.8

itc = 1.0 - tc
X_src_class0 = X_valid[Y_valid == src_class0].numpy()
X_src_class1 = X_valid[Y_valid == src_class1].numpy()
X_dst_class = X_valid[Y_valid == dst_class].numpy()

len_src0 = int(tc * len(X_src_class0))
len_src1 = int(tc * len(X_src_class1))
len_dst = int(tc * len(X_dst_class))

ilen_src0 = int(itc * len(X_src_class0))
ilen_src1 = int(itc * len(X_src_class1))
ilen_dst = int(itc * len(X_dst_class))

np.random.seed(42)
np.random.shuffle(X_src_class0)
np.random.shuffle(X_src_class1)

X_src_class0[:(len_src0 // 2)] = decoder.predict(Utils.encoded(X_src_class0[:(len_src0 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[:(len_src1 // 2)] = decoder.predict(Utils.encoded(X_src_class1[:(len_src1 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])

X_src_class0[-(ilen_src0 // 2):] = decoder.predict(Utils.encoded(X_src_class0[-(ilen_src0 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[-(ilen_src1 // 2):] = decoder.predict(Utils.encoded(X_src_class1[-(ilen_src1 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])

"""random_indices = np.random.choice(X_src_class0[-ilen_src0:].shape[0], 100, replace=False)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_src_class0[-ilen_src0:][random_indices[i]], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()"""

true_labels = np.array([src_class0, src_class1, dst_class])

classifier = load_model("./Models/Classifieur/classifier.keras")
res_classifier = load_model(f"./Models/Classifieur/residual-classifier-128-{src_class0}{src_class1}{dst_class}.keras")
detect_classifier = load_model(f"./Models/Classifieur/residual-detection-classifier-128-{src_class0}{src_class1}{dst_class}.keras")

X_classes = np.concatenate((X_src_class0[-ilen_src0:], X_src_class1[-ilen_src1:], X_dst_class[-ilen_dst:]))
X_classes = tf.image.resize(X_classes, (28, 28))
Y_classes = to_categorical(np.concatenate((np.full(ilen_src0, 0), np.full(ilen_src1, 1), np.full(ilen_dst, 2))), 3)
Y_classes_detect = np.concatenate((np.full(ilen_src0 - ilen_src0 // 2, 0), np.full(ilen_src0 // 2, 1), np.full(ilen_src1 - ilen_src1 // 2, 0), np.full(ilen_src1 // 2, 1), np.full(ilen_dst, 0)))

X_classes_translated = np.concatenate((X_src_class0[-(ilen_src0 // 2):], X_src_class1[-(ilen_src1 // 2):]))
X_classes_translated = tf.image.resize(X_classes_translated, (28, 28))
Y_classes_translated = to_categorical(np.concatenate((np.full(ilen_src0 // 2, 0), np.full(ilen_src1 // 2, 1))), 3)

X_classes_unchanged = np.concatenate((X_src_class0[-ilen_src0:-(ilen_src0 // 2)], X_src_class1[-ilen_src1:-(ilen_src1 // 2)]))
X_classes_unchanged = tf.image.resize(X_classes_unchanged, (28, 28))
Y_classes_unchanged = to_categorical(np.concatenate((np.full(ilen_src0 - ilen_src0 // 2, 0), np.full(ilen_src1 - ilen_src1 // 2, 1))), 3)

X_classes_full = np.concatenate((X_src_class0, X_src_class1, X_dst_class))
X_classes_full = tf.image.resize(X_classes_full, (28, 28))
Y_classes_real_full = to_categorical(np.concatenate((
    np.full(len_src0 // 2, dst_class),
    np.full(len(X_src_class0) - len_src0 // 2 - ilen_src0 // 2, src_class0),
    np.full(ilen_src0 // 2, dst_class),
    
    np.full(len_src1 // 2, dst_class),
    np.full(len(X_src_class1) - len_src1 // 2 - ilen_src1 // 2, src_class1),
    np.full(ilen_src1 // 2, dst_class),

    np.full(len(X_dst_class), dst_class),
)), 10)

X_classes_translated_full = np.concatenate((X_src_class0[:(len_src0 // 2)], X_src_class0[-(ilen_src0 // 2):], X_src_class1[:(len_src1 // 2)], X_src_class1[-(ilen_src1 // 2):]))
X_classes_translated_full = tf.image.resize(X_classes_translated_full, (28, 28))
Y_classes_translated_full = to_categorical((np.full(len_src0 // 2 + ilen_src0 // 2 + len_src1 // 2 + ilen_src1 // 2, dst_class)), 10)

labels = list(range(10))

Y_pred = classifier.predict(X_classes_full)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_real_full, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm_full = confusion_matrix(Y_true_classes, Y_pred_classes, labels=labels)

cm_full = cm_full[true_labels]

row_sums = cm_full.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm_full / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=[src_class0, src_class1, dst_class], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Classification des chiffres source et translatés ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-normal-classifier-full-confusion-{src_class0}{src_class1}{dst_class}.png")








Y_pred = classifier.predict(X_classes_translated_full)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_translated_full, axis = 1)

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=labels)

cm = cm[true_labels]

dst_index = np.where(true_labels == dst_class)[0][0]
cm = np.vstack((cm_full[dst_index] - cm[dst_index], cm[dst_index, :]))

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=["Non translatés", "Translatés"], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.suptitle(f"Classification des {dst_class} ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-normal-classifier-translated-dst-confusion-{src_class0}{src_class1}{dst_class}.png")


Y_pred = res_classifier.predict(X_classes)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm_full = confusion_matrix(Y_true_classes, Y_pred_classes)

labels = [src_class0, src_class1, dst_class]

row_sums = cm_full.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm_full / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres source et translatés ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-classifier-confusion-{src_class0}{src_class1}{dst_class}.png")



Y_pred = res_classifier.predict(X_classes_translated)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_translated, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=[0, 1, 2])
cm = cm[:-1, :]

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=[src_class0, src_class1, dst_class], yticklabels=[src_class0, src_class1], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres translatés uniquement ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-translated-classifier-confusion-{src_class0}{src_class1}{dst_class}.png")





Y_pred = res_classifier.predict(X_classes_unchanged)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_unchanged, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=[0, 1, 2])
cm = cm[:-1, :]

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])


plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=[src_class0, src_class1, dst_class], yticklabels=[src_class0, src_class1], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres inchangés uniquement ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-unchanged-classifier-confusion-{src_class0}{src_class1}{dst_class}.png")



Y_pred = detect_classifier.predict(X_classes)

Y_pred_classes = (Y_pred >= 0.5).astype(int)

accuracy = accuracy_score(Y_classes_detect, Y_pred_classes)

average_certainty = 1.0 - np.mean(np.abs(Y_pred - Y_pred_classes))

cm = confusion_matrix(Y_classes_detect, Y_pred_classes)

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

detection_labels = ["Non détecté", "Détecté"]

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=detection_labels, yticklabels=detection_labels, vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection de la translation ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-detection-translation-confusion-{src_class0}{src_class1}{dst_class}.png")



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

predicted = Utils.encoded(image, "", encoder, decoder, 2, batch_size, False)

src_class, p, linp = Utils.classify(image, classifier)

if src_class == src_class0 or src_class == src_class1:
    src_class_g, p_g, linp_g = Utils.classify(image, res_classifier)

    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    axes[0].imshow(cv2.cvtColor(image64, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(image[0], cmap="gray")
    axes[1].set_title("Image seuillée")
    axes[1].axis("off")

    axes[1].text(0.5, -0.15, f"({src_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[1].transAxes)
    axes[1].text(0.5, -0.3, f"({true_labels[src_class_g]}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[1].transAxes)

    decoded = decoder.predict(predicted, batch_size = batch_size)

    axes[2].imshow(decoded[0], cmap="gray")
    axes[2].set_title("Reconstruction")
    axes[2].axis("off")

    guessed_class, p, linp = Utils.classify(decoded, classifier)
    guessed_class_g, p_g, linp_g = Utils.classify(decoded, res_classifier)
    axes[2].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[2].transAxes)
    axes[2].text(0.5, -0.3, f"({true_labels[guessed_class_g]}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[2].transAxes)

    translated = predicted + encoded_means[dst_class] - encoded_means[src_class]
    translated_decoded = decoder.predict(translated)

    axes[3].imshow(translated_decoded[0], cmap="gray")
    axes[3].set_title("Translaté")
    axes[3].axis("off")

    guessed_class, p, linp = Utils.classify(translated_decoded, classifier)
    guessed_class_g, p_g, linp_g = Utils.classify(translated_decoded, res_classifier)
    axes[3].text(0.5, -0.15, f"({guessed_class}, {p.max():.3f})", fontsize=14, color="blue", ha="center", transform=axes[3].transAxes)
    axes[3].text(0.5, -0.3, f"({true_labels[guessed_class_g]}, {p_g.max():.3f})", fontsize=14, color="red", ha="center", transform=axes[3].transAxes)

    plt.tight_layout()
    plt.savefig(f"./Results/mnist-trace-translated-image-{src_class0}{src_class1}{dst_class}.png")

"""Y_tsne = np.concatenate((
    np.full(ilen_src0 - ilen_src0 // 2, 0),
    np.full(ilen_src0 // 2, 1),
    np.full(ilen_src1 - ilen_src1 // 2, 2),
    np.full(ilen_src1 // 2, 3),
    np.full(ilen_dst, 4),
))"""

X_classes_full = tf.image.resize(X_classes_full, (64, 64))
X_encoded_classes = encoder.predict(X_classes_full)

Y_tsne = np.concatenate((
    np.full(len_src0 // 2, 1),
    np.full(len(X_src_class0) - len_src0 // 2 - ilen_src0 // 2, 0),
    np.full(ilen_src0 // 2, 1),

    np.full(len_src1 // 2, 3),
    np.full(len(X_src_class1) - len_src1 // 2 - ilen_src1 // 2, 2),
    np.full(ilen_src1 // 2, 3),

    np.full(len(X_dst_class), 4),
))

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)

if src_class == src_class0 or src_class == src_class1:
    X_encoded = np.concatenate((X_encoded_classes, predicted, translated))
    X_tsne = tsne.fit_transform(X_encoded)
    X_predicted = X_tsne[-2]
    X_translated = X_tsne[-1]
    X_tsne = X_tsne[:-2]
else:
    X_tsne = tsne.fit_transform(X_encoded_classes)

plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=Y_tsne,
    cmap="Paired",
    alpha=0.35,
    s=40
)

unique_classes = np.unique(Y_tsne)
norm = Normalize(vmin = min(unique_classes), vmax = max(unique_classes))
labels = [
    f"{src_class0} inchangé",
    f"{src_class0} translaté en {dst_class}",
    f"{src_class1} inchangé",
    f"{src_class1} translaté en {dst_class}",
    f"{dst_class} inchangé"
]
for i, label in enumerate(labels):
    plt.scatter([], [], color=plt.cm.Paired(norm(i)), label=label)

if src_class == src_class0 or src_class == src_class1:
    plt.scatter(X_predicted[0], X_predicted[1], marker="+", color="red", s=150, label="Image source")
    plt.scatter(X_translated[0], X_translated[1], marker="+", color="blue", s=150, label="Image translatée")

    plt.arrow(X_predicted[0], X_predicted[1],
            X_translated[0] - X_predicted[0],
            X_translated[1] - X_predicted[1],
            color="purple", width=0.01, head_width=0.2, length_includes_head=True)

plt.title(f"t-SNE : ({src_class0}, {src_class1}) → {dst_class}")
plt.legend()
plt.tight_layout()
plt.savefig(f"./Results/mnist-translation-res-full-tsne-{src_class0}{src_class1}{dst_class}.png")