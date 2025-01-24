import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import math

from sklearn.manifold import TSNE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow.keras.backend as K # backend différent de keras.backend

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

num_epochs = 6

img_shape = (28, 28, 1)
batch_size = 16

latent_dims = [8, 32, 64, 128]

X_valid_encoded = [None] * len(latent_dims)
X_decoded = [None] * len(latent_dims)

decoder = [None] * len(latent_dims)

for li in range(len(latent_dims)):
    latent_dim = latent_dims[li]

    input_img = Input(shape=img_shape)

    # Encoder
    x = Conv2D(32, 3, padding="same")(input_img)
    x = ReLU()(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = ReLU()(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = ReLU()(x)

    shape_before_flattening = K.int_shape(x)

    x = Flatten()(x)

    z = Dense(latent_dim)(x)

    # Decoder
    decoder_input = Input(K.int_shape(z)[1:])

    x = Dense(np.prod(shape_before_flattening[1:]))(decoder_input)
    x = ReLU()(x)

    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv2DTranspose(128, 3, padding="same")(x)
    x = ReLU()(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, 3, padding="same")(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, 3, padding="same")(x)
    x = ReLU()(x)

    x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    decoder[li] = Model(decoder_input, x)

    z_decoded = decoder[li](z)

    autoencoder = Model(input_img, z_decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()

    autoencoder.fit(
        x=X_train,
        y=X_train,
        shuffle=True, 
        epochs=num_epochs, 
        batch_size=batch_size, 
        validation_data=(X_valid, X_valid)
    )

    encoder = Model(input_img, z)
    X_valid_encoded[li] = encoder.predict(X_valid, batch_size=batch_size)

    X_decoded[li] = decoder[li].predict(X_valid_encoded[li], batch_size=batch_size)

    encoder.save("./Models/AE/mnist-" + str(latent_dim) + "-encoder.keras")
    decoder[li].save("./Models/AE/mnist-" + str(latent_dim) + "-decoder.keras")