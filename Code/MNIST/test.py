import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

# Find the indices of all "7"s in the validation set
indices_of_sevens = np.where(Y_valid != 700)[0]

# Select the first 100 "7"s
sevens = X_valid[indices_of_sevens[:100]]

# Plot the 100 digits in a 10x10 grid
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
axes = axes.ravel()  # Flatten the axes array

for i in np.arange(100):
    axes[i].imshow(sevens[i], cmap='gray')
    axes[i].axis('off')  # Turn off axis

plt.subplots_adjust(wspace=0.5)
plt.show()
