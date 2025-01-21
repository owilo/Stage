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
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda
from keras.models import Model
from keras import metrics
#from tensorflow.python.keras.backend import get_session
import tensorflow.keras.backend as K

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_train = X_train.reshape(-1,28,28,1)

X_valid = X_valid.astype('float32') / 255.
X_valid = X_valid.reshape(-1,28,28,1)

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 8

input_img = Input(shape=img_shape)

x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)

def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mu + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mu, z_log_sigma])

decoder_input = Input(K.int_shape(z)[1:])

x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)

x = Reshape(shape_before_flattening[1:])(x)

x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
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

num_epochs = 4

vae.fit(x=X_train, y=None, shuffle=True, epochs=num_epochs, batch_size=batch_size, validation_data=(X_valid, None))

encoder = Model(input_img, z_mu)
X_valid_encoded = encoder.predict(X_valid, batch_size=batch_size)

"""tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X_valid_encoded)

plt.figure(figsize=(8, 8))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=2, cmap=colormaps.get_cmap('Paired'))
plt.title("t-SNE")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./Results/latent_tsne.png")"""

plt.figure(figsize=(10, 10))
plt.scatter(X_valid_encoded[:, 0], X_valid_encoded[:, 1], c=Y_valid, cmap='brg')
plt.colorbar()
plt.savefig("./Results/latent_projection.png")

n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        Z_sample = np.array([[xi, yi]])
        Z_sample = np.tile(Z_sample, batch_size).reshape(batch_size, 2)
        X_decoded = decoder.predict(Z_sample, batch_size=batch_size)
        digit = X_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.savefig("./Results/latent_translation.png")