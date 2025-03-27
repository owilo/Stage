import tensorflow as tf
from tensorflow import keras

from Code.Models import AAE
from Code.Utils import cache, utils

# Make sure that the module with your custom classes (AAE, AAEEncoder, etc.) is imported.
# For example:
# from Code.Models.aae_model import AAE

MODEL_PATH = cache.MODEL_FOLDER / "AAE" / "aae16.keras"
vae = keras.models.load_model(MODEL_PATH)


import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST test dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

# Generate reconstructions
reconstructions = vae.predict(x_test)

# Plot a few original and reconstructed images
n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    ax.axis("off")

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructions[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()

# Get latent codes (you might need to call the encoder directly)
encoder = vae.encoder
z_mean_class, z_log_var_class, z_class, z_mean_style, z_log_var_style, z_style = encoder(x_test)

# Example: Visualize the first two dimensions of the class latent codes
plt.figure(figsize=(8, 6))
plt.scatter(z_class[:, 0], z_class[:, 1], c=y_test, cmap="viridis", s=5)
plt.colorbar()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Latent Space Visualization (Class Component)")
plt.show()

# Select a single image and get its latent representation
sample_image = x_test[0:1]
z_mean_class, z_log_var_class, z_class, z_mean_style, z_log_var_style, z_style = encoder(sample_image)

# Generate a set of style codes by sampling randomly around the original style latent code
num_variations = 10
random_styles = z_style + 0.5 * tf.random.normal((num_variations, z_style.shape[-1]))

# Repeat the fixed class latent code for each variation
repeated_class = tf.repeat(z_class, repeats=num_variations, axis=0)

# Generate reconstructions by combining fixed class with varied style
reconstructed_variations = vae.decoder((repeated_class, random_styles))

# Plot the variations
plt.figure(figsize=(20, 2))
for i in range(num_variations):
    ax = plt.subplot(1, num_variations, i + 1)
    plt.imshow(reconstructed_variations[i].numpy().squeeze(), cmap="gray")
    ax.axis("off")
plt.suptitle("Variations in Reconstruction by Varying the Style Latent Code", fontsize=16)
plt.show()
