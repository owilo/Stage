import numpy as np
import pandas as pd

from keras.datasets import mnist
from sklearn.manifold import TSNE
from matplotlib import colormaps
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten
X = np.zeros((x_train.shape[0], 784))

for i in range(x_train.shape[0]):
    X[i] = x_train[i].flatten()

X = pd.DataFrame(X)
Y = pd.DataFrame(y_train)

df = X

# t-SNE
tsne = TSNE(max_iter = 1000)
tsne_results = tsne.fit_transform(X.values)

df["label"] = Y

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1, title="t-SNE")

s = ax.scatter(
    x = tsne_results[:, 0], 
    y = tsne_results[:, 1], 
    c = df["label"].squeeze(), 
    cmap = colormaps.get_cmap("Paired")
)

sorted_labels = sorted(df["label"].unique())

handles, _ = s.legend_elements(prop = "colors", num = len(sorted_labels))
legend_labels = [str(int(label)) for label in sorted_labels]
ax.legend(handles, legend_labels)

plt.savefig("./Results/mnist_tsne.png")