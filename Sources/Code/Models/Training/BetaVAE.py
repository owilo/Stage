import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from Code.Models.Common.layers import Sampling
from Code.Utils import utils, models

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
    
    def requires_labels(self):
        return False

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
        ))
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
    
BetaVAE.Encoder = Encoder
BetaVAE.Decoder = Decoder
BetaVAE.Sampling = Sampling

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    x_train = tf.image.resize(x_train, (64, 64))
    x_test = tf.image.resize(x_test, (64, 64))

    parser = argparse.ArgumentParser(description="BetaVAE")
    parser.add_argument("-l", type=int, default=128, help="Taille du vecteur latent")
    parser.add_argument("-e", type=int, default=5, help="Nombre d'époques")
    parser.add_argument("-b", type=int, default=32, help="Taille de batch")
    parser.add_argument("--ds", type=float, default=1.0, help="Taille du dataset (0 à 1), 1 inclut aussi le dataset de test")
    parser.add_argument("--beta", type=float, default=6.0, help="Coefficient β de pondération pour la régularisation")

    args = parser.parse_args()

    latent_dim = args.l
    num_epochs = args.e
    batch_size = args.b
    dataset_size = max(0.0, min(args.ds), 1.0)
    beta = args.beta

    print(f">> l : {args.l}, β : {args.beta}, e : {num_epochs}, b : {batch_size}")

    vae = BetaVAE(latent_dim=latent_dim, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam())

    if (dataset_size < 1.0):
        print(f">> Taille du dataset d'entraînement : {dataset_size}")

        x_train_left, _, _, _ = utils.split_dataset(x_train, y_train, dataset_size)
        vae.fit(
            x_train_left,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_split=0.1,
            validation_batch_size=batch_size
        )
    else:
        print(">> Entraînement classique")
        vae.fit(
            x_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=x_test,
            validation_batch_size=batch_size
        )

    dummy_x = np.random.rand(1, 64, 64, 1).astype("float32")
    _ = vae(dummy_x)

    filename = f"betavae-{latent_dim}.keras" if dataset_size == 1 else f"h-betavae-{latent_dim}.keras"

    model_definition = {
        "type": "autoencoder",
        "category": "BetaVAE",
        "file": filename,
        "input_shape": [64, 64, 1],
        "output_shape": [64, 64, 1],
        "latent_shape": [latent_dim],
        "labels": False,
        "dataset_range": [0, dataset_size]
    }

    models.save_model(vae, model_definition)