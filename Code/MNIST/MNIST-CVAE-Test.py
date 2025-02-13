import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Custom Layers and Sampling Layer
# -------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

import tensorflow as tf
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    def __init__(self, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.dense = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x, y = inputs
        x = self.conv1(x)  # Shape: (14,14,32)
        x = self.conv2(x)  # Shape: (7,7,64)
        x = self.flatten(x)  # 7*7*64=3136
        x = self.concat([x, y])  # 3136 + 10 = 3146
        x = self.dense(x)  # 256 units
        return self.z_mean(x), self.z_log_var(x), self.sampling((self.z_mean(x), self.z_log_var(x)))

class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate()
        self.dense = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7,7,64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        z, y = inputs
        x = self.concat([z, y])
        x = self.dense(x)  # 7*7*64=3136
        x = self.reshape(x)  # (7,7,64)
        x = self.deconv1(x)  # (14,14,64)
        x = self.deconv2(x)  # (28,28,32)
        return self.conv_out(x)  # (28,28,1)

# -------------------------------
# CVAE Model (with custom train/test steps)
# -------------------------------

class CVAE(keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()
        # Track losses
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        # data is a tuple: (x, y)
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder((x, y))
            x_recon = self.decoder((z, y))
            # Use binary crossentropy per pixel and sum over image dimensions
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder((x, y))
        x_recon = self.decoder((z, y))

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
            )
        )
        beta = 35.0
        kl_loss = beta * (-0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        ))
        total_loss = reconstruction_loss + kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        # For inference: given (x, y) returns the reconstruction
        x, y = inputs
        z_mean, z_log_var, z = self.encoder((x, y))
        return self.decoder((z, y))

# -------------------------------
# Data Preparation
# -------------------------------

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)  # shape: (num_samples, 28, 28, 1)
x_test = x_test.astype("float32") / 255.
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot vectors (10 classes)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# -------------------------------
# Model Compilation and Training
# -------------------------------

latent_dim = 32
cvae = CVAE(latent_dim=latent_dim)
cvae.compile(optimizer=keras.optimizers.Adam())

# Train the CVAE model
cvae.fit(
    x=x_train, y=y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Testing: Translate a Source Digit into All Classes
# -------------------------------

# Select a source image (say, the first test sample)
x0 = x_test[0:1]        # shape (1, 28, 28, 1)
y0 = y_test[0:1]        # its original one-hot label

# Obtain the latent representation (we ignore the mean here, but you could use z_mean)
z_mean, z_log_var, z = cvae.encoder((x0, y0))

# For visualization: create a 2-row subplot grid.
# Top row: decoding the same latent z with different target labels (translation)
# Bottom row: display the original x0 for comparison.
# Provided indices for one source test image per digit class
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

# Create an 11 (rows) x 10 (columns) subplot grid.
# Row 0: original source image.
# Rows 1-10: translation of the latent vector with target labels 0...9.
fig, axs = plt.subplots(11, 10, figsize=(10, 11))

for j, idx in enumerate(digits):
    # Get the source test image and its one-hot label (as provided in y_test)
    x_src = x_test[idx:idx+1]   # shape (1, 28, 28, 1)
    y_src = y_test[idx:idx+1]   # shape (1, 10)
    
    # Compute the latent representation using the source label.
    z_mean, z_log_var, z = cvae.encoder((x_src, y_src))
    
    # Row 0: display the original image
    axs[0, j].imshow(x_src[0, :, :, 0], cmap='gray')
    axs[0, j].axis('off')
    axs[0, j].set_title(f"Idx {idx}\nOrig", fontsize=10)
    
    # For each target class (0-9), decode the same latent vector with the new one-hot label.
    for i in range(10):
        # Create one-hot vector for target class i.
        y_target = np.zeros((1, num_classes), dtype="float32")
        y_target[0, i] = 1.0
        
        # Decode using the same latent vector but with target label.
        x_decoded = cvae.decoder((z, y_target))
        
        axs[i+1, j].imshow(x_decoded[0, :, :, 0], cmap='gray')
        axs[i+1, j].axis('off')
        
        # Optionally, label the leftmost column rows with the target digit.
        if j == 0:
            axs[i+1, j].set_ylabel(str(i), fontsize=12)

plt.suptitle("CVAE: Source Test Digits and Their Translations into All 10 Classes", fontsize=16)
plt.tight_layout()
plt.show()
