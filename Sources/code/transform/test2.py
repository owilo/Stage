import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from code.utils import models

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

num_classes = 10
examples = []
labels = []

for i in range(num_classes):
    idx = np.where(y_train == i)[0][0]
    examples.append(x_train[idx])
    labels.append(y_train[idx])

examples = np.stack(examples, axis=0)

one_hot_labels = tf.one_hot(labels, depth=10)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "category": "BranchVAE"}
))

_, _, z, _, _, w = autoencoder.encoder.predict(examples)
decoded = autoencoder.decoder.predict([z, w])

plt.figure(figsize=(20, 4))
for i in range(num_classes):
    ax = plt.subplot(2, num_classes, i + 1)
    plt.imshow(examples[i].squeeze(), cmap="gray")
    plt.axis("off")

    ax = plt.subplot(2, num_classes, i + 1 + num_classes)
    plt.imshow(decoded[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.suptitle("Originals (top) and Reconstructions (bottom)")
plt.savefig("fig3.png", dpi=150)
plt.close()

style_swap_images = []

for i in range(num_classes):
    row_images = []
    for j in range(num_classes):
        z_style = z[j:j+1]
        w_class = w[i:i+1]
        swapped = autoencoder.decoder.predict([z_style, w_class])
        row_images.append(swapped[0])
    style_swap_images.append(row_images)

style_swap_images = np.array(style_swap_images)

fig, axs = plt.subplots(num_classes, num_classes, figsize=(20, 20))
for i in range(num_classes):
    for j in range(num_classes):
        axs[i, j].imshow(style_swap_images[i, j].squeeze(), cmap="gray")
        axs[i, j].axis("off")

plt.suptitle("Style Swap Grid (rows: w_class, columns: z_style)")
plt.savefig("fig4.png", dpi=150)
plt.close()
