import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import seaborn as sns

from keras.datasets import mnist

import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical

import itertools

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

tc = 0.75

split_index = [int(tc * len(cls)) for cls in X_classes]

X_classes2 = [cls[idx:] for cls, idx in zip(X_classes, split_index)]

encoder = load_model("./Models/DISVAE/mnist-128-h-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-h-decoder.keras")

encoded_means = utils.encoded_means(X_split1, Y_split1, "h_encoded_means_disvae", encoder, decoder, 2, 32)

Y_classes_translated2 = np.array([])
Y_classes_isTranslated2 = np.array([])

for src_class in range(10):
    src_classes = np.array_split(X_classes2[src_class], 10)

    for dst_class in range(10):
        print(src_class, dst_class)
        Y_classes_translated2 = np.append(Y_classes_translated2, np.full(len(src_classes[dst_class]), dst_class))
        Y_classes_isTranslated2 = np.append(Y_classes_isTranslated2, np.full(len(src_classes[dst_class]), int(src_class != dst_class)))

        if src_class == dst_class:
            continue

        X_encoded_src = utils.encoded(src_classes[dst_class], "", encoder, decoder, 3, 32, False)
        translation = encoded_means[dst_class] - encoded_means[src_class]
        X_translated = X_encoded_src + translation
        src_classes[dst_class] = decoder.predict(X_translated, batch_size = 32)

    X_classes2[src_class] = np.concatenate(src_classes)

Y_classes2 = np.repeat(np.arange(10), np.array([len(src_class) for src_class in X_classes2]))
Y_classes2 = to_categorical(Y_classes2, 10)

X_classes2 = np.array(list(itertools.chain(*X_classes2)))
X_classes2 = tf.image.resize(X_classes2, (28, 28))

indices = np.arange(X_classes2.shape[0])
np.random.shuffle(indices)
indices = tf.convert_to_tensor(indices, dtype=tf.int32)
X_classes2 = tf.gather(X_classes2, indices)
Y_classes2 = tf.gather(Y_classes2, indices)
Y_classes_translated2 = tf.gather(Y_classes_translated2, indices)
Y_classes_isTranslated2 = tf.gather(Y_classes_isTranslated2, indices)

classifier = load_model("./Models/Classifieur/classifier.keras")
res_classifier = load_model("./Models/Classifieur/residual-classifier-128.keras")
detect_classifier = load_model("./Models/Classifieur/residual-detection-classifier-128.keras")

Y_pred = res_classifier.predict(X_classes2)

Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_classes2, axis = 1)

certainty = np.max(Y_pred, axis=1)
average_certainty = np.mean(certainty)

accuracy = accuracy_score(Y_true_classes, Y_pred_classes)

cm = confusion_matrix(Y_true_classes, Y_pred_classes)

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
plt.savefig(f"./Results/mnist-trace-classifier-confusion.png")

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
plt.suptitle(f"Classification ", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig(f"./Results/mnist-trace-normal-classifier-confusion.png")


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
plt.savefig(f"./Results/mnist-trace-detection-translation-confusion.png")