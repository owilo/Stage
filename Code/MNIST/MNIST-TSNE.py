import numpy as np
import pandas as pd

from keras.datasets import mnist
from sklearn.manifold import TSNE
from matplotlib import colormaps
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten
X = np.zeros((x_train.shape[0], 784))

for i in range(x_train.shape[0]):
    X[i] = x_train[i].flatten()

X = pd.DataFrame(X)
Y = pd.DataFrame(y_train)

df = X

# t-sne
tsne = TSNE(max_iter = 300)
tsne_results = tsne.fit_transform(X.values)

df["label"] = Y

fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title="t-SNE")

s = ax.scatter(
    x = tsne_results[:,0], 
    y = tsne_results[:,1], 
    c = df["label"], 
    cmap = colormaps.get_cmap("Paired")
)

handles, labels = s.legend_elements(prop = "colors", num = len(df["label"].unique()))
legend_labels = [str(int(label)) for label in df["label"].unique()]
ax.legend(handles, legend_labels, title="Labels")

plt.savefig("./Results/mnist_tsne.png")