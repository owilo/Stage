import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from code.models.common.layers import Sampling
from code.utils import utils, models

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim=2, num_classes=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # convolutional feature extractor
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        # VAE heads
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()
        # classification head
        self.classifier = layers.Dense(num_classes, activation='softmax', name='y_pred')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        # VAE outputs
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        # classifier output
        y_pred = self.classifier(x)
        return z_mean, z_log_var, z, y_pred

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
        })
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
        return super(Decoder, self).get_config()

    def requires_labels(self):
        return True

@tf.keras.utils.register_keras_serializable()
class SCVAE(keras.Model):
    def __init__(
        self,
        latent_dim=2,
        num_classes=10,
        class_loss_weight=1.0,
        final_beta=4.0,
        annealing_steps=6000,
        **kwargs
    ):
        super(SCVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_loss_weight = class_loss_weight
        self.final_beta = final_beta
        self.annealing_steps = annealing_steps
        # replace encoder with classifier head
        self.encoder = Encoder(latent_dim=latent_dim, num_classes=num_classes)
        self.decoder = Decoder()
        # metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.classification_accuracy = keras.metrics.CategoricalAccuracy(name="classification_accuracy")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.classification_loss_tracker,
            self.classification_accuracy,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, y_pred = self.encoder(x, training=True)
            # reconstruction
            x_recon = self.decoder((z, y))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
                )
            )
            # KL divergence
            step = tf.cast(self.optimizer.iterations, tf.float32)
            beta = self.final_beta * tf.minimum(1.0, step / self.annealing_steps)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            ) * beta
            # classification loss
            class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, y_pred))
            # combined loss
            total_loss = reconstruction_loss + kl_loss + self.class_loss_weight * class_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.classification_accuracy.update_state(y, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.classification_loss_tracker.result(),
            "class_accuracy": self.classification_accuracy.result(),
            "beta": beta,
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z, y_pred = self.encoder(x, training=False)
        x_recon = self.decoder((z, y))
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        ) * self.final_beta
        class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, y_pred))
        total_loss = reconstruction_loss + kl_loss + self.class_loss_weight * class_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.classification_accuracy.update_state(y, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.classification_loss_tracker.result(),
            "class_accuracy": self.classification_accuracy.result(),
        }

    def call(self, inputs, training=False):
        x, y = inputs
        z_mean, z_log_var, z, y_pred = self.encoder(x, training=training)
        x_recon = self.decoder((z, y))
        return x_recon

    def get_config(self):
        config = super(SCVAE, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'class_loss_weight': self.class_loss_weight,
            'final_beta': self.final_beta,
            'annealing_steps': self.annealing_steps,
        })
        return config

# Usage example omitted for brevity, follows same pattern as CVAE but instantiates SCVAE


# alias for saving/loading consistency
SCVAE.Encoder = Encoder
SCVAE.Decoder = Decoder
SCVAE.Sampling = Sampling

if __name__ == "__main__":
    # reproducibility
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    # load & preprocess MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    # one-hot labels
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    parser = argparse.ArgumentParser(description="SCVAE")
    parser.add_argument("-l", type=int, default=32, help="latent vector size")
    parser.add_argument("-e", type=int, default=80, help="epochs")
    parser.add_argument("-b", type=int, default=32, help="batch size")
    parser.add_argument("--ds", type=float, default=1.0, help="dataset fraction")
    parser.add_argument("--beta", type=float, default=4.0, help="β weighting")
    parser.add_argument("--ans", type=int, default=27500, help="annealing steps")
    parser.add_argument("--clw", type=float, default=1.0, help="class loss weight")
    parser.add_argument("--name", type=str, default="scvae", help="model name")
    args = parser.parse_args()

    # instantiate
    scvae = SCVAE(
        latent_dim=args.l,
        num_classes=10,
        class_loss_weight=args.clw,
        final_beta=args.beta,
        annealing_steps=args.ans
    )
    scvae.compile(optimizer=keras.optimizers.Adam())

    # training split
    if args.ds < 1.0:
        x_tr, y_tr, _, _ = utils.split_dataset(x_train, y_train, args.ds)
        scvae.fit(
            x_tr,
            y_tr,
            epochs=args.e,
            batch_size=args.b,
            validation_split=0.1
        )
    else:
        scvae.fit(
            x_train,
            y_train,
            epochs=args.e,
            batch_size=args.b,
            validation_data=(x_test, y_test)
        )

    # sanity-check forward pass
    dummy_x = np.random.rand(1, 28, 28, 1).astype("float32")
    dummy_y = tf.one_hot([3], depth=10)
    _ = scvae((dummy_x, dummy_y))

    model_def = {
        "type": "autoencoder",
        "category": "SCVAE",
        "name": args.name,
        "input_shape": [28, 28, 1],
        "output_shape": [28, 28, 1],
        "latent_shape": [args.l],
        "labels": True,
        "dataset_range": [0, args.ds]
    }
    models.save_model(scvae, model_def)