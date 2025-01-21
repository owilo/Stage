import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps

from scipy.stats import norm
from sklearn.manifold import TSNE

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import metrics
#from tensorflow.python.keras.backend import get_session
import tensorflow.keras.backend as K

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_train = X_train.reshape(-1,28,28,1)
#X_train = X_train[:6000]
#Y_train = Y_train[:6000]

X_valid = X_valid.astype('float32') / 255.
X_valid = X_valid.reshape(-1,28,28,1)
#X_valid = X_valid[:1000]
#Y_valid = Y_valid[:1000]

num_epochs = 8

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 8

input_img = Input(shape=img_shape)

# Encoder
x = Conv2D(32, 3, padding='same')(input_img)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(64, 3, padding='same', strides=(2, 2))(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

"""x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)"""

x = Conv2D(64, 3, padding='same')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

shape_before_flattening = K.int_shape(x)

x = Flatten()(x)
x = Dense(32)(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)

def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mu + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mu, z_log_sigma])

# Decoder
decoder_input = Input(K.int_shape(z)[1:])

x = Dense(np.prod(shape_before_flattening[1:]))(decoder_input)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Reshape(shape_before_flattening[1:])(x)

x = Conv2DTranspose(64, 3, padding='same')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

"""x = Conv2DTranspose(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)"""

x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(32, 3, padding='same')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)

z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x, z_decoded, z_mu, z_log_sigma = inputs
        loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
        self.add_loss(loss)
        return x

y = CustomVariationalLayer()([input_img, z_decoded, z_mu, z_log_sigma])

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

vae.fit(x = X_train, y = None, shuffle = True, epochs = num_epochs, batch_size = batch_size, validation_data = (X_valid, None))

encoder = Model(input_img, z_mu)
X_valid_encoded = encoder.predict(X_valid, batch_size = batch_size)

# t-SNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_valid_encoded_tSNE = tsne.fit_transform(X_valid_encoded)

plt.figure(figsize=(8, 8))

for i in range(10):
    indices = (Y_valid == i)
    plt.scatter(
        X_valid_encoded_tSNE[indices, 0],
        X_valid_encoded_tSNE[indices, 1],
        label=str(i),
        cmap=colormaps.get_cmap("Paired")
    )

plt.legend()
plt.title("t-SNE (latent)")
plt.tight_layout()
plt.savefig("./Results/mnist_tsne_latent_space.png")

# Average

means = [None] * 10
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    means[i] = np.mean(X_valid_encoded[Y_valid == i], axis = 0).reshape(1, -1)

    X_decoded = decoder.predict(means[i], batch_size=batch_size)

    axes[i].imshow(X_decoded[0])
    axes[i].set_title(str(i))
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist_average.png")

# Translation

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(10):
    for j in range(10):
        selected_data = X_valid_encoded[Y_valid == i][0]
        X_decoded = decoder.predict(selected_data + means[j] - means[i], batch_size=batch_size)
        
        ax = axes[i, j]
        ax.imshow(X_decoded[0].reshape(28, 28))
        ax.axis('off')

plt.tight_layout()
plt.savefig("./Results/mnist_translate_means.png")