import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix

from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from Code.Training.Classifier import Classifier
from Code.Training.TraceClassifier import TraceClassifier
from Code.Training.TraceDetector import TraceDetector
from Code.Utils import cache, latent, utils

def compute_confusion_matrix(cm, certainties, labels, filename, title_prefix=""):
    accuracy = np.trace(cm) / np.sum(cm)
    avg_certainty = np.mean(certainties)

    percentages = (cm / np.sum(cm, axis=1)) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{percentages[i, j]:.1f}%"

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=100.0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.xlabel("Classe prédite", fontsize=12)
    plt.ylabel("Classe cible", fontsize=12)
    plt.suptitle(title_prefix, fontsize=18)
    plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {avg_certainty:.2%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion '{filename}' sauvegardée")

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

x_train_l, y_train_l, x_train_r, y_train_r = utils.split_dataset(x_train, y_train, 0.5) # Moitié gauche pour le VAE

x_train_r = np.concatenate((x_train_r, x_test))
y_train_r = np.concatenate((y_train_r, y_test))

_, _, x_train_rr, y_train_rr = utils.split_dataset(x_train_r, y_train_r, 0.75) # 75% de gauche pour l'entraînement des classifieurs

x_src, y_src, y_dst = utils.split_src_to_dst(x_train_rr, y_train_rr)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "h-betavae128.keras")
classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "classifier.keras")
trace_classifier = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "trace-classifier.keras")
trace_detector = tf.keras.models.load_model(cache.MODEL_FOLDER / "Classifier" / "trace-detector.keras")

z_src = latent.encode_n(
    autoencoder,
    x=x_src,
    y=y_src,
    n=3,
    save_cache=True
)

z_class_distributions = latent.class_distributions_n(
    autoencoder,
    x=x_train_l,
    y=y_train_l,
    n=2,
    save_cache=True
)

z_translated = latent.translate(z_src, y_src, y_dst, z_class_distributions)
x_decoded = autoencoder.decoder.predict(z_translated)

x_decoded, z_src, z_translated, y_src, y_dst = utils.shuffle(x_decoded, z_src, z_translated, y_src, y_dst)

y_src_categorical = keras.utils.to_categorical(y_src, 10)
y_dst_categorical = keras.utils.to_categorical(y_dst, 10)
y_trans = (y_src == y_dst).astype(int)

#
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