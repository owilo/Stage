import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from MNIST_CVAE_Train import Sampling, Encoder, Decoder, CVAE # Important

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype("float32") / 255.
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

cvae = tf.keras.models.load_model("./Models/CVAE/cvae16.keras")


digits = [
    1333,  # 0
    9415,  # 1
    3773,  # 2
    524,   # 3
    1980,  # 4
    1874,  # 5
    4252,  # 6
    6960,  # 7
    8466,  # 8
    5333   # 9
]

fig, axs = plt.subplots(10, 11, figsize=(11, 10))

for j, idx in enumerate(digits):
    x_src = x_test[idx:idx + 1]
    y_src = y_test[idx:idx + 1]
    
    _, _, z = cvae.encoder(x_src)
    
    axs[j, 0].imshow(x_src[0, :, :, 0], cmap='gray')
    axs[j, 0].axis('off')
    
    for i in range(10):
        y_target = np.zeros((1, num_classes), dtype="float32")
        y_target[0, i] = 1.0
        
        x_decoded = cvae.decoder((z, y_target))
        
        axs[j, i + 1].imshow(x_decoded[0, :, :, 0], cmap='gray')
        axs[j, i + 1].axis('off')

plt.suptitle("CVAE", fontsize=16)
plt.tight_layout()
plt.savefig("./Results/mnist-cvae.png")

z_mean, _, _ = cvae.encoder(x_test)
z_mean = z_mean.numpy()

tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z_mean)

labels = np.argmax(y_test, axis=1)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="Paired", alpha=0.35)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE CVAE (style)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("./Results/mnist-cvae-style-tsne.png")

z_mean_class = np.concatenate((z_mean, y_test.astype(float)), axis=1)

tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z_mean_class)

labels = np.argmax(y_test, axis=1)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="Paired", alpha=0.35)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE CVAE")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("./Results/mnist-cvae-tsne.png")