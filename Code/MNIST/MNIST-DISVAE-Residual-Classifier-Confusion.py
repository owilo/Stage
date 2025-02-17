import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix

import tensorflow.keras.backend as K

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

len_src0 = int(itc * len(X_src_class0))
len_src1 = int(itc * len(X_src_class1))
len_dst = int(itc * len(X_dst_class))

np.random.seed(42)
np.random.shuffle(X_src_class0)
np.random.shuffle(X_src_class1)

X_src_class0[-(len_src0 // 2):] = decoder.predict(utils.encoded(X_src_class0[-(len_src0 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[-(len_src1 // 2):] = decoder.predict(utils.encoded(X_src_class1[-(len_src1 // 2):], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])


random_indices = np.random.choice(X_src_class0[-len_src0:].shape[0], 100, replace=False)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_src_class0[-len_src0:][random_indices[i]], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()



X_classes = np.concatenate((X_src_class0[-len_src0:], X_src_class1[-len_src1:], X_dst_class[-len_dst:]))
X_classes = tf.image.resize(X_classes, (28, 28))
Y_classes = to_categorical(np.concatenate((np.full(len_src0, 0), np.full(len_src1, 1), np.full(len_dst, 2))), 3)

X_classes_translated = np.concatenate((X_src_class0[-(len_src0 // 2):], X_src_class1[-(len_src1 // 2):]))
X_classes_translated = tf.image.resize(X_classes_translated, (28, 28))
Y_classes_translated = to_categorical(np.concatenate((np.full(len_src0 // 2, 0), np.full(len_src1 // 2, 1))), 3)



classifier = load_model("./Models/Classifieur/residual-classifier-128.keras")

Y_pred = classifier.predict(X_classes)

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
sns.heatmap(cm, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels)
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres source et translatés ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-classifier-confusion.png")



Y_pred = classifier.predict(X_classes_translated)

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
sns.heatmap(cm, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels[:-1])
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle(f"Détection des traces sur des chiffres translatés uniquement ({src_class0}, {src_class1}) → {dst_class}", fontsize=18)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig("./Results/mnist-trace-translated-classifier-confusion.png")