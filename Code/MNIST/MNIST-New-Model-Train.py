import numpy as np

from sklearn.manifold import TSNE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow as tf
import tensorflow.keras.backend as K # backend différent de keras.backend

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

num_epochs = 5

img_shape = (64, 64, 1)
batch_size = 16

latent_dim = 128

input_img = Input(shape = img_shape)

# Encoder
x = Conv2D(128, 3, padding="same", activation="relu")(input_img)
x = MaxPooling2D(2, padding="same")(x)
x = Conv2D(128, 3, padding="same", activation="relu")(x)
x = MaxPooling2D(2, padding="same")(x)
x = Conv2D(64, 3, padding="same", activation="relu")(x)
x = MaxPooling2D(2, padding="same")(x)
x = Conv2D(32, 3, padding="same", activation="relu")(x)
x = MaxPooling2D(2, padding="same")(x)

shape_before_flattening = K.int_shape(x)

x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dense(256, activation="relu")(x)

z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)

def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape = (K.shape(z_mu)[0], latent_dim), mean = 0.0, stddev = 1.0)
    return z_mu + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mu, z_log_sigma])

# Decoder
decoder_input = Input(K.int_shape(z)[1:])

x = Dense(256, activation="relu")(decoder_input)
x = Dense(256, activation="relu")(x)
x = Dense(np.prod(shape_before_flattening[1:]), activation="relu")(x)
x = Reshape(shape_before_flattening[1:])(x)
x = Conv2DTranspose(32, 3, padding="same", activation="relu")(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(64, 3, padding="same", activation="relu")(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(128, 3, padding="same", activation="relu")(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(128, 3, padding="same", activation="relu")(x)
x = UpSampling2D(2)(x)
x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)

decoder = Model(decoder_input, x)

z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        xent_loss *= 64 * 64
        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x, z_decoded, z_mu, z_log_sigma = inputs
        loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
        self.add_loss(loss)
        return x

y = CustomVariationalLayer()([input_img, z_decoded, z_mu, z_log_sigma])

encoder = Model(input_img, z_mu)
encoder.summary()
decoder.summary()

vae = Model(input_img, y)
vae.compile(optimizer = "adam", loss = None)
#vae.summary()

# Fitting

vae.fit(x = X_train, y = None, shuffle = True, epochs = num_epochs, batch_size = batch_size, validation_data = (X_valid, None))

encoder.save("./Models/VAE/mnist-" + str(latent_dim) + "-encoder-dis2.keras")
decoder.save("./Models/VAE/mnist-" + str(latent_dim) + "-decoder-dis2.keras")