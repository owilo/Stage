import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K

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
n_data = X_train.shape[0]

input_img = Input(shape=img_shape)
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
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mu + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mu, z_log_sigma])

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
decoder = Model(decoder_input, x, name=f"btcvaedecoder-{latent_dim}")
z_decoded = decoder(z)

def log_density_gaussian(x, mu, logvar):
    norm = -0.5 * (tf.math.log(2.0 * np.pi) + logvar)
    log_dens = norm - 0.5 * ((x - mu) ** 2 / tf.exp(logvar))
    return log_dens

def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = tf.unstack(tf.shape(x))
    x = tf.reshape(x, [batch_size, 1, 1, dim])
    mu = tf.reshape(mu, [1, batch_size, 1, dim])
    logvar = tf.reshape(logvar, [1, batch_size, 1, dim])
    log_dens = log_density_gaussian(x, mu, logvar)
    log_dens = tf.squeeze(log_dens, axis=2)
    return log_dens

def log_importance_weight_matrix(batch_size, dataset_size):
    N = tf.cast(dataset_size, tf.float32)
    M = tf.cast(batch_size - 1, tf.float32)
    strat_weight = (N - M) / (N * M)
    W = tf.ones((batch_size, batch_size)) * strat_weight
    diag = tf.eye(batch_size) * (1.0 / N - strat_weight)
    W += diag
    return tf.math.log(W)

def linear_annealing(init, fin, step, annealing_steps):
    return tf.cond(
        tf.equal(annealing_steps, 0),
        lambda: fin,
        lambda: tf.minimum(init + (fin - init) * step / annealing_steps, fin)
    )

class BtcvaeLossLayer(keras.layers.Layer):
    def __init__(self, n_data, alpha=1.0, beta=1.0, gamma=1.0, is_mss=True, steps_anneal=0, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_mss = is_mss
        self.steps_anneal = steps_anneal
        self.n_train_steps = self.add_weight(name='n_train_steps', shape=(), initializer='zeros', trainable=False)

    def call(self, inputs, training=None):
        x, recon_x, z_mu, z_log_var, z = inputs
        
        x_flat = K.flatten(x)
        recon_x_flat = K.flatten(recon_x)
        rec_loss = K.sum(K.binary_crossentropy(x_flat, recon_x_flat)) * (64 * 64)
        
        log_pz = log_density_gaussian(z, tf.zeros_like(z), tf.zeros_like(z))
        log_pz = K.sum(log_pz, axis=1)
        
        log_q_zCx = log_density_gaussian(z, z_mu, z_log_var)
        log_q_zCx = K.sum(log_q_zCx, axis=1)
        
        mat_log_qz = matrix_log_density_gaussian(z, z_mu, z_log_var)
        if self.is_mss:
            log_iw_mat = log_importance_weight_matrix(tf.shape(z)[0], self.n_data)
            mat_log_qz += log_iw_mat[..., tf.newaxis]
        
        log_qz = tf.reduce_logsumexp(K.sum(mat_log_qz, axis=2), axis=1)
        log_prod_qzi = K.sum(tf.reduce_logsumexp(mat_log_qz, axis=1), axis=1)
        
        mi_loss = K.mean(log_q_zCx - log_qz)
        tc_loss = K.mean(log_qz - log_prod_qzi)
        dw_kl_loss = K.mean(log_prod_qzi - log_pz)
        
        anneal_reg = linear_annealing(0.0, 1.0, self.n_train_steps, self.steps_anneal)
        total_loss = rec_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * anneal_reg * dw_kl_loss
        
        self.add_loss(total_loss)
        
        if training:
            self.n_train_steps.assign_add(1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

loss_layer = BtcvaeLossLayer(n_data=n_data, steps_anneal=1000)
outputs = loss_layer([input_img, z_decoded, z_mu, z_log_sigma, z])
vae = Model(input_img, outputs)
vae.compile(optimizer='adam')

encoder = Model(input_img, z_mu, name=f"btcvaeencoder-{latent_dim}")
encoder.summary()
decoder.summary()

vae.fit(X_train, None, shuffle=True, epochs=num_epochs, batch_size=batch_size, validation_data=(X_valid, None))

encoder.save(f"./Models/DISVAE/mnist-btcvae2-{latent_dim}-encoder.keras")
decoder.save(f"./Models/DISVAE/mnist-btcvae2-{latent_dim}-decoder.keras")