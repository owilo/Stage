import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from code.models.common.layers import Sampling
from code.utils import utils, models

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
        # class latent
        self.zc_mean = layers.Dense(latent_dim)
        self.zc_log_var = layers.Dense(latent_dim)
        # style latent
        self.zs_mean = layers.Dense(latent_dim)
        self.zs_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        zc_mean = self.zc_mean(x)
        zc_log_var = self.zc_log_var(x)
        zc = self.sampling((zc_mean, zc_log_var))
        zs_mean = self.zs_mean(x)
        zs_log_var = self.zs_log_var(x)
        zs = self.sampling((zs_mean, zs_log_var))
        return zc_mean, zc_log_var, zc, zs_mean, zs_log_var, zs

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

    def call(self, z):
        x = self.concat(z)
        x = self.dense0(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.conv_out(x)

    def get_config(self):
        return super(Decoder, self).get_config()

@tf.keras.utils.register_keras_serializable()
class BranchVAE(keras.Model):
    def __init__(self,
                 latent_dim=2,
                 num_classes=10,
                 alpha_rec=1.0,
                 beta_s=1.0,
                 gamma_class=1.0,
                 **kwargs):
        super(BranchVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()
        # mixture prior parameters
        self.mu_prior = self.add_weight(
            shape=(num_classes, latent_dim), initializer='random_normal', trainable=True, name='mu_prior')
        self.logvar_prior = self.add_weight(
            shape=(num_classes, latent_dim), initializer='zeros', trainable=True, name='logvar_prior')
        # classification head
        self.classifier = layers.Dense(num_classes, activation='softmax')
        self.alpha_rec = alpha_rec
        self.beta_s = beta_s
        self.gamma_class = gamma_class
        # metrics
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_c_tracker = keras.metrics.Mean(name="kl_c")
        self.kl_s_tracker = keras.metrics.Mean(name="kl_s")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_accuracy = keras.metrics.CategoricalAccuracy(name="class_acc")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_c_tracker,
            self.kl_s_tracker,
            self.class_loss_tracker,
            self.class_accuracy
        ]

    def _log_normal(self, z, mean, logvar):
        # z: [B,D], mean/logvar: [B,D] or [K,D]
        return -0.5 * (tf.square(z - mean) / tf.exp(logvar) + logvar + tf.math.log(2.0 * np.pi))

    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            zc_mean, zc_log_var, zc, zs_mean, zs_log_var, zs = self.encoder(x)
            # reconstruction
            x_recon = self.decoder([zc, zs])
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1,2))
            )
            # KL style
            kl_s = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + zs_log_var - tf.square(zs_mean) - tf.exp(zs_log_var), axis=1)
            )
            # KL class via mixture prior (Monte Carlo)
            # log q(zc)
            log_q = tf.reduce_sum(self._log_normal(zc, zc_mean, zc_log_var), axis=1)
            # log p(zc) via mixture
            # expand for components
            zc_exp = tf.expand_dims(zc, 1)  # [B,1,D]
            mu_exp = tf.expand_dims(self.mu_prior, 0)  # [1,K,D]
            lv_exp = tf.expand_dims(self.logvar_prior, 0)  # [1,K,D]
            log_p_comp = tf.reduce_sum(self._log_normal(zc_exp, mu_exp, lv_exp), axis=2)  # [B,K]
            log_p = tf.math.reduce_logsumexp(log_p_comp - tf.math.log(float(self.num_classes)), axis=1)
            kl_c = tf.reduce_mean(log_q - log_p)
            # classification
            pred = self.classifier(zc_mean)
            class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, pred))
            # total
            total_loss = (
                self.alpha_rec * rec_loss
                + self.beta_s * kl_s
                + kl_c
                + self.gamma_class * class_loss
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_c_tracker.update_state(kl_c)
        self.kl_s_tracker.update_state(kl_s)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        zc_mean, zc_log_var, zc, zs_mean, zs_log_var, zs = self.encoder(x)
        x_recon = self.decoder([zc, zs])
        rec_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1,2))
        )
        kl_s = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + zs_log_var - tf.square(zs_mean) - tf.exp(zs_log_var), axis=1)
        )
        # kl_c as in train
        log_q = tf.reduce_sum(self._log_normal(zc, zc_mean, zc_log_var), axis=1)
        zc_exp = tf.expand_dims(zc, 1)
        mu_exp = tf.expand_dims(self.mu_prior, 0)
        lv_exp = tf.expand_dims(self.logvar_prior, 0)
        log_p_comp = tf.reduce_sum(self._log_normal(zc_exp, mu_exp, lv_exp), axis=2)
        log_p = tf.math.reduce_logsumexp(log_p_comp - tf.math.log(float(self.num_classes)), axis=1)
        kl_c = tf.reduce_mean(log_q - log_p)
        pred = self.classifier(zc_mean)
        class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, pred))
        total_loss = (
            self.alpha_rec * rec_loss
            + self.beta_s * kl_s
            + kl_c
            + self.gamma_class * class_loss
        )
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_c_tracker.update_state(kl_c)
        self.kl_s_tracker.update_state(kl_s)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        zc_mean, zc_log_var, zc, zs_mean, zs_log_var, zs = self.encoder(inputs)
        return self.decoder([zc, zs])

    def get_config(self):
        config = super(BranchVAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "alpha_rec": self.alpha_rec,
            "beta_s": self.beta_s,
            "gamma_class": self.gamma_class,
            "num_classes": self.num_classes
        })
        return config

# backward compatibility
BranchVAE.Encoder = Encoder
BranchVAE.Decoder = Decoder
BranchVAE.Sampling = Sampling

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=int, default=16)
    parser.add_argument("-e", type=int, default=10)
    parser.add_argument("-b", type=int, default=32)
    parser.add_argument("--alpha_rec", type=float, default=1.0)
    parser.add_argument("--beta_s", type=float, default=2.0)
    parser.add_argument("--gamma_class", type=float, default=2.0)
    parser.add_argument("--ds", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="branchvae")
    args = parser.parse_args()
    
    if args.ds < 1.0:
        x_train, y_train, _, _ = utils.split_dataset(x_train, y_train, args.ds)

    model = BranchVAE(latent_dim=args.l,
                      num_classes=10,
                      alpha_rec=args.alpha_rec,
                      beta_s=args.beta_s,
                      gamma_class=args.gamma_class)
    model.compile(optimizer=keras.optimizers.Adam())

    model.fit(
        x_train, y_train,
        epochs=args.e,
        batch_size=args.b,
        validation_split=0.1 if args.ds != 1.0 else None,
        validation_data=(x_test, y_test) if args.ds == 1.0 else None,
    )

    _ = model(x_test[:1])
    model_def = {
        "type": "autoencoder",
        "category": "BranchVAE",
        "name": args.name,
        "input_shape": [28,28,1],
        "output_shape": [28,28,1],
        "latent_shape": [args.l],
        "labels": False,
        "dataset_range": [0, args.ds],
    }
    models.save_model(model, model_def)