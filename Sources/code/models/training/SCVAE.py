import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from code.models.common.layers import Sampling
from code.utils import utils, models

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')

        self.flatten = layers.Flatten()
        self.dense   = layers.Dense(256, activation='relu')

        self.z_mean    = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling  = Sampling()

        self.classifier = layers.Dense(num_classes, activation='softmax', name='y_pred')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = self.dense(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))

        y_pred = self.classifier(x)
        return z_mean, z_log_var, z, y_pred

    def get_config(self):
        return super(Encoder, self).get_config()


@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate()

        self.dense0 = layers.Dense(256, activation='relu')

        self.dense1 = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7,7,64))

        self.conv1  = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.conv2  = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')

        self.out = layers.Conv2D(1, 3, padding='same', activation='sigmoid')

    def call(self, inputs, training=False):
        z, y = inputs
        x = self.concat([z, y])
        x = self.dense0(x)
        x = self.reshape(self.dense1(x))
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x)

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
        ssim_weight=1.0,
        **kwargs
    ):
        super(SCVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_loss_weight = class_loss_weight
        self.final_beta = final_beta
        self.annealing_steps = annealing_steps
        self.ssim_weight = ssim_weight

        self.encoder = Encoder(latent_dim=latent_dim, num_classes=num_classes)
        self.decoder = Decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.classification_accuracy = keras.metrics.CategoricalAccuracy(name="classification_accuracy")
        self.ssim_loss_tracker = keras.metrics.Mean(name="ssim_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.classification_loss_tracker,
            self.classification_accuracy,
            self.ssim_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, y_pred = self.encoder(x, training=True)
            x_recon = self.decoder((z, y), training=True)

            bce_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
                )
            )

            ssim_val = tf.reduce_mean(tf.image.ssim(x, x_recon, max_val=1.0))
            ssim_loss = 1.0 - ssim_val

            reconstruction_loss = bce_loss + self.ssim_weight * ssim_loss

            step = tf.cast(self.optimizer.iterations, tf.float32)
            beta = self.final_beta * tf.minimum(1.0, step / self.annealing_steps)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            ) * beta

            class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, y_pred))

            total_loss = reconstruction_loss + kl_loss + self.class_loss_weight * class_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.classification_accuracy.update_state(y, y_pred)
        self.ssim_loss_tracker.update_state(ssim_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.classification_loss_tracker.result(),
            "class_accuracy": self.classification_accuracy.result(),
            "ssim_loss": self.ssim_loss_tracker.result(),
            "beta": beta,
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z, y_pred = self.encoder(x, training=False)
        x_recon = self.decoder((z, y), training=False)

        bce_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
            )
        )
        ssim_val = tf.reduce_mean(tf.image.ssim(x, x_recon, max_val=1.0))
        ssim_loss = 1.0 - ssim_val
        reconstruction_loss = bce_loss + self.ssim_weight * ssim_loss

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
        self.ssim_loss_tracker.update_state(ssim_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.classification_loss_tracker.result(),
            "class_accuracy": self.classification_accuracy.result(),
            "ssim_loss": self.ssim_loss_tracker.result(),
        }

    def call(self, inputs, training=False):
        x, y = inputs
        z_mean, z_log_var, z, y_pred = self.encoder(x, training=training)
        return self.decoder((z, y), training=training)

    def get_config(self):
        config = super(SCVAE, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'class_loss_weight': self.class_loss_weight,
            'final_beta': self.final_beta,
            'annealing_steps': self.annealing_steps,
            'ssim_weight': self.ssim_weight,
        })
        return config

SCVAE.Encoder = Encoder
SCVAE.Decoder = Decoder
SCVAE.Sampling = Sampling

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    parser = argparse.ArgumentParser(description="SCVAE")
    parser.add_argument("-l", type=int, default=32, help="latent vector size")
    parser.add_argument("-e", type=int, default=100, help="epochs")
    parser.add_argument("-b", type=int, default=16, help="batch size")
    parser.add_argument("--ds", type=float, default=1.0, help="dataset fraction")
    parser.add_argument("--beta", type=float, default=3.0, help="β weighting")
    parser.add_argument("--ans", type=int, default=30000, help="annealing steps")
    parser.add_argument("--clw", type=float, default=1.0, help="class loss weight")
    parser.add_argument("--ssimw", type=float, default=3.0, help="SSIM loss weight")
    parser.add_argument("--name", type=str, default="scvae", help="model name")
    args = parser.parse_args()

    scvae = SCVAE(
        latent_dim=args.l,
        num_classes=10,
        class_loss_weight=args.clw,
        final_beta=args.beta,
        annealing_steps=args.ans,
        ssim_weight=args.ssimw
    )
    scvae.compile(optimizer=keras.optimizers.Adam())

    if args.ds < 1.0:
        x_train, y_train, x_test, y_test = utils.split_dataset(x_train, y_train, args.ds)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(args.b)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.b)

    scvae.fit(
        train_ds,
        epochs=args.e,
        validation_data=test_ds
    )

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