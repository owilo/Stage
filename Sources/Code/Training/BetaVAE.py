import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

from Code.Utils import cache, utils

@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_resize = layers.Resizing(64, 64)
        self.conv1 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPooling2D(2, padding="same")
        self.conv2 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool2 = layers.MaxPooling2D(2, padding="same")
        self.conv3 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.pool3 = layers.MaxPooling2D(2, padding="same")
        self.conv4 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.pool4 = layers.MaxPooling2D(2, padding="same")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(256, activation="relu")
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.input_resize(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, shape_before_flattening=(4, 4, 32), latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.shape_before_flattening = shape_before_flattening
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(256, activation="relu")
        self.dense3 = layers.Dense(np.prod(shape_before_flattening), activation="relu")
        self.reshape_layer = layers.Reshape(shape_before_flattening)
        self.deconv1 = layers.Conv2DTranspose(32, 3, padding="same", activation="relu")
        self.upsample1 = layers.UpSampling2D(2)
        self.deconv2 = layers.Conv2DTranspose(64, 3, padding="same", activation="relu")
        self.upsample2 = layers.UpSampling2D(2)
        self.deconv3 = layers.Conv2DTranspose(128, 3, padding="same", activation="relu")
        self.upsample3 = layers.UpSampling2D(2)
        self.deconv4 = layers.Conv2DTranspose(128, 3, padding="same", activation="relu")
        self.upsample4 = layers.UpSampling2D(2)
        self.conv_out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape_layer(x)
        x = self.deconv1(x)
        x = self.upsample1(x)
        x = self.deconv2(x)
        x = self.upsample2(x)
        x = self.deconv3(x)
        x = self.upsample3(x)
        x = self.deconv4(x)
        x = self.upsample4(x)
        return self.conv_out(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "shape_before_flattening": self.shape_before_flattening,
            "latent_dim": self.latent_dim,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class BetaVAE(tf.keras.Model):
    def __init__(self, latent_dim=128, beta=6.0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(shape_before_flattening=(4, 4, 32), latent_dim=latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = self.beta * (-0.5 * tf.reduce_mean(
                tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            ))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = self.beta * (-0.5 * tf.reduce_mean(
            tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )) # or reduce_sum?
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "beta": self.beta,
        })
        return config

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    x_train = tf.image.resize(x_train, (64, 64))
    x_test = tf.image.resize(x_test, (64, 64))

    latent_dim = 128
    beta = 6.0
    num_epochs = 5
    batch_size = 32

    vae = BetaVAE(latent_dim=latent_dim, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(
        x_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=x_test,
        validation_batch_size=batch_size
    )

    dummy_x = np.random.rand(1, 64, 64, 1).astype("float32")
    _ = vae(dummy_x)

    MODEL_PATH = cache.MODEL_FOLDER / "BetaVAE"
    os.makedirs(MODEL_PATH, exist_ok=True)
    vae.save(os.path.join(MODEL_PATH, "betavae128.keras"))