import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
import sys

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
    def __init__(self, latent_dim=2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.concat = layers.Concatenate()
        self.dense0 = layers.Dense(256, activation='relu')
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        z, y = inputs
        x = self.concat([z, y])
        x = self.dense0(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.conv_out(x)

    def get_config(self):
        config = super(Decoder, self).get_config()
        return config
    
    def requires_labels(self):
        return True

@tf.keras.utils.register_keras_serializable()
class CVAE(keras.Model):
    def __init__(self, latent_dim=2, final_beta=4.0, annealing_steps=6000, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.final_beta = final_beta
        self.annealing_steps = annealing_steps
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()
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
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            x_recon = self.decoder((z, y))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
                )
            )
            step = tf.cast(self.optimizer.iterations, tf.float32)
            beta = self.final_beta * tf.minimum(1.0, step / self.annealing_steps)
            kl_loss = beta * (-0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            ))
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
            "beta": beta
        }
    
    @tf.function
    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder((z, y))
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
            )
        )
        beta = self.final_beta
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
        x, y = inputs
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder((z, y))
    
    def get_config(self):
        config = super(CVAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "final_beta": self.final_beta,
            "annealing_steps": self.annealing_steps,
        })
        return config

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    latent_dim = 128
    num_epochs = 30
    batch_size = 32

    cvae = CVAE(latent_dim=latent_dim, final_beta=3.5, annealing_steps=27500)
    cvae.compile(optimizer=keras.optimizers.Adam())

    if (len(sys.argv) > 1):
        p = max(0.0, min(float(sys.argv[1]), 1.0))
        print(f">> Taille du dataset d'entraînement : {p}")

        x_train_left, _, _, _ = utils.split_dataset(x_train, y_train, p)
        cvae.fit(
            x_train_left,
            epochs=math.ceil(num_epochs / p),
            batch_size=batch_size,
            validation_split=0.1,
            validation_batch_size=batch_size
        )
    else:
        print(">> Entraînement classique")
        cvae.fit(
            x_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=x_test,
            validation_batch_size=batch_size
        )

    dummy_x = np.random.rand(1, 28, 28, 1).astype("float32")
    dummy_y = np.zeros((1, 10)).astype("float32")
    _ = cvae((dummy_x, dummy_y))

    MODEL_PATH = cache.MODEL_FOLDER / "CVAE"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    cvae.save(MODEL_PATH / "cvae16_2.keras")