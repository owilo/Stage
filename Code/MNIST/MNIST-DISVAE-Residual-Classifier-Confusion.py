import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import seaborn as sns

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

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

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

src_class0 = 0
src_class1 = 1
dst_class = 2

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

X_src_class0[:(len_src0 // 2)] = decoder.predict(utils.encoded(X_src_class0[:(len_src0 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[:(len_src1 // 2)] = decoder.predict(utils.encoded(X_src_class1[:(len_src1 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])

X_src_class0[-(ilen_src0 // 2):] = decoder.predict(utils.encoded(X_src_class0[-(ilen_src0 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[-(ilen_src1 // 2):] = decoder.predict(utils.encoded(X_src_class1[-(ilen_src1 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])

"""random_indices = np.random.choice(X_src_class0[-ilen_src0:].shape[0], 100, replace=False)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_src_class0[-ilen_src0:][random_indices[i]], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()"""

classifier = load_model("./Models/Classifieur/classifier.keras")
res_classifier = load_model("./Models/Classifieur/residual-classifier-128.keras")

X_classes = np.concatenate((X_src_class0[-ilen_src0:], X_src_class1[-ilen_src1:], X_dst_class[-ilen_dst:]))
X_classes = tf.image.resize(X_classes, (28, 28))
Y_classes = to_categorical(np.concatenate((np.full(ilen_src0, 0), np.full(ilen_src1, 1), np.full(ilen_dst, 2))), 3)

X_classes_translated = np.concatenate((X_src_class0[-(ilen_src0 // 2):], X_src_class1[-(ilen_src1 // 2):]))
X_classes_translated = tf.image.resize(X_classes_translated, (28, 28))
Y_classes_translated = to_categorical(np.concatenate((np.full(ilen_src0 // 2, 0), np.full(ilen_src1 // 2, 1))), 3)

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

cm_full = cm_full[~np.all(cm_full == 0, axis=1)]

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
plt.savefig("./Results/mnist-trace-normal-classifier-full-confusion.png")








Y_pred = classifier.predict(X_classes_translated_full)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_translated_full, axis = 1)

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=labels)

cm = cm[~np.all(cm == 0, axis=1)]

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=labels)
cm = np.vstack((cm_full[dst_class] - cm[2], cm[2, :]))

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=False, yticklabels=["Non translatés", "Translatés"], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.suptitle(f"Classification des {dst_class} ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-normal-classifier-translated-dst-confusion.png")


Y_pred = res_classifier.predict(X_classes)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm = confusion_matrix(Y_true_classes, Y_pred_classes)

labels = [src_class0, src_class1, dst_class]

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres source et translatés ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion.png")



Y_pred = res_classifier.predict(X_classes_translated)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes_translated, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=labels)
cm = cm[:-1, :]

row_sums = cm.sum(axis=1, keepdims=True)
percentages = np.where(row_sums == 0, 0, cm / row_sums * 100)

annot = np.array([["{:.2f}%".format(val) for val in row] for row in percentages])

plt.figure(figsize=(10, 8))
sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels[:-1], vmin=0.0, vmax=100.0)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres translatés uniquement ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-translated-classifier-confusion.png")


X_classes = tf.image.resize(X_classes, (64, 64))
X_encoded_classes = encoder.predict(X_classes)

Y_tsne = np.concatenate((
    np.full(ilen_src0 - ilen_src0 // 2, 0),
    np.full(ilen_src0 // 2, 1),
    np.full(ilen_src1 - ilen_src1 // 2, 2),
    np.full(ilen_src1 // 2, 3),
    np.full(ilen_dst, 4),
))

tsne = TSNE(n_components = 2, random_state = 1337, max_iter = 300)
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

plt.title(f"t-SNE : ({src_class0}, {src_class1}) → {dst_class}")
plt.legend()
plt.tight_layout()
plt.savefig("./Results/mnist-translation-res-tsne.png")