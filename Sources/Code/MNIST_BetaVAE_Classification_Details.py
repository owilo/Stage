import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from keras.datasets import mnist

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

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
classifier = load_model("./Models/Classifieur/classifier.keras")

X_reencoded_valid = Utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)
encoded_means = Utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

total_conf_matrix = np.zeros((10, 10), dtype=int)

global_certainties = []
global_categories = []

for src_class in range(10):
    digits = X_reencoded_valid[Y_valid == src_class]
    mean_encoded_src = encoded_means[src_class]
    
    for dst_class in range(10):
        if src_class == dst_class:
            continue

        mean_encoded_dst = encoded_means[dst_class]
        translation = mean_encoded_dst - mean_encoded_src
        translated = digits + translation

        decoded = decoder.predict(translated, batch_size=batch_size)
        decoded = tf.image.resize(decoded, (28, 28)).numpy()

        Y_pred_proba = classifier.predict(decoded)
        guessed_classes = np.argmax(Y_pred_proba, axis=1)
        certainty = np.max(Y_pred_proba, axis=1)
        
        for i, guess in enumerate(guessed_classes):
            if guess == dst_class:
                category = "Cible"
            elif guess == src_class:
                category = "Source"
            else:
                category = "Autre"
            global_certainties.append(certainty[i])
            global_categories.append(category)

global_certainties = np.array(global_certainties)
global_categories = np.array(global_categories)

bins = np.array([0.0, 0.8, 0.9, 0.95, 0.99, 1.0])
bin_labels = ["< 80%", "80% - 90%", "90% - 95%", "95% - 99%", "99% - 100%"]

categories_list = ["Cible", "Source", "Autre"]
counts = {cat: np.zeros(len(bins)-1) for cat in categories_list}

for i in range(len(bins)-1):
    bin_mask = (global_certainties >= bins[i]) & (global_certainties < bins[i+1])
    for cat in categories_list:
        counts[cat][i] = np.sum((global_categories == cat) & bin_mask)

overall_counts = {cat: np.sum(counts[cat]) for cat in categories_list}
total = sum(overall_counts.values())
overall_percentages = {cat: (overall_counts[cat] / total) * 100 for cat in categories_list}

total_samples = sum(sum(counts[cat]) for cat in categories_list)

counts_percent = {cat: (counts[cat] / total_samples) * 100 for cat in categories_list}

stacked_heights = np.sum([counts_percent[cat] for cat in categories_list], axis=0)
max_bar_height = np.max(stacked_heights) 
y_max = np.ceil(max_bar_height / 5) * 5

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(bins) - 1)

colors = {"Cible": "#4daf4a", "Source": "#e41a1c", "Autre": "#377eb8"}

for cat in categories_list:
    ax.bar(bin_labels, counts_percent[cat], bottom=bottom, color=colors[cat])
    bottom += counts_percent[cat]

ax.set_xlabel("Intervalle de certitude")
ax.set_ylabel("% d'images translatées")
ax.set_title("Distribution de la certitude de la prédiction")

ax.set_yticks(np.arange(0, y_max + 5, 5))
ax.set_ylim(0, y_max)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

axins = inset_axes(ax, width=2.25, height=2.25, loc="upper left", bbox_to_anchor=(0.05, 0.95), bbox_transform=ax.transAxes, borderpad=0)

pie_colors = [colors[cat] for cat in categories_list]
axins.pie([overall_percentages[cat] for cat in categories_list],
          labels=[f'{cat} ({overall_percentages[cat]:.2f}%)' for cat in categories_list],
          colors=pie_colors, startangle=90)
axins.set_title("Classification", fontsize=14, y=0.95)

plt.tight_layout()
plt.savefig("./Results/mnist-translation-classification-details.png")