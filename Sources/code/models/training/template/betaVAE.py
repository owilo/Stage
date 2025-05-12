import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from code.models.common.layers import Sampling

@tf.keras.utils.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        z = self.sampling((mean, log_var))
        return mean, log_var, z

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class Decoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense0 = layers.Dense(256, activation='relu')
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, z):
        x = self.dense0(z)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.conv_out(x)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class BetaVAE(keras.Model):
    def __init__(self, latent_dim=2, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.rec_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.rec_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(x)
            x_recon = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
            )
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, _ = data
        mean, log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        )
        total_loss = recon_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        config = super().get_config()
        config.update({'latent_dim': self.latent_dim, 'beta': self.beta})
        return config

if __name__ == '__main__':
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent_dim', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()

    model = BetaVAE(latent_dim=args.latent_dim, beta=args.beta)
    model.compile(optimizer=keras.optimizers.Adam())

    model.fit(
        x_train, x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, x_test)
    )