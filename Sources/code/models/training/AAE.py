import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from code.models.common.layers import Sampling, GradientReversal
from code.utils import cache, utils

@tf.keras.utils.register_keras_serializable()
class AAEEncoder(tf.keras.Model):
    def __init__(self, latent_dim_class=16, latent_dim_style=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim_class = latent_dim_class
        self.latent_dim_style = latent_dim_style
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        self.z_mean_class = layers.Dense(latent_dim_class)
        self.z_log_var_class = layers.Dense(latent_dim_class)
        self.z_mean_style = layers.Dense(latent_dim_style)
        self.z_log_var_style = layers.Dense(latent_dim_style)
        self.sampling = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        # class
        z_mean_class = self.z_mean_class(x)
        z_log_var_class = self.z_log_var_class(x)
        z_class = self.sampling((z_mean_class, z_log_var_class))
        # style
        z_mean_style = self.z_mean_style(x)
        z_log_var_style = self.z_log_var_style(x)
        z_style = self.sampling((z_mean_style, z_log_var_style))
        return (z_mean_class, z_log_var_class, z_class,
                z_mean_style, z_log_var_style, z_style)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim_class": self.latent_dim_class,
            "latent_dim_style": self.latent_dim_style,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class AAEDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate()
        self.dense0 = layers.Dense(256, activation='relu')
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        z_class, z_style = inputs
        x = self.concat([z_class, z_style])
        x = self.dense0(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.conv_out(x)

    def get_config(self):
        config = super().get_config()
        return config
    
    def requires_labels(self):
        return False

@tf.keras.utils.register_keras_serializable()
class Classifier(tf.keras.Model):
    def __init__(self, latent_dim_class, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim_class": self.latent_dim_class})
        return config

@tf.keras.utils.register_keras_serializable()
class AdvClassifier(tf.keras.Model):
    def __init__(self, latent_dim_style, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim_style": self.latent_dim_style})
        return config

@tf.keras.utils.register_keras_serializable()
class AAE(tf.keras.Model):
    def __init__(self, latent_dim_class=16, latent_dim_style=16, 
                 beta_class=1.0, beta_style=0.0, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim_class = latent_dim_class
        self.latent_dim_style = latent_dim_style
        self.beta_class = beta_class
        self.beta_style = beta_style
        self.gamma = gamma
        self.encoder = AAEEncoder(latent_dim_class=latent_dim_class, 
                                 latent_dim_style=latent_dim_style)
        self.decoder = AAEDecoder()
        self.classifier = Classifier(latent_dim_class=latent_dim_class, num_classes=10)
        self.adv_classifier = AdvClassifier(latent_dim_style=latent_dim_style, num_classes=10)
        self.grl = GradientReversal(lambda_=1.0)

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.class_loss_tracker,
                self.adv_loss_tracker]

    def call(self, inputs):
        z_mean_class, z_log_var_class, z_class, z_mean_style, z_log_var_style, z_style = self.encoder(inputs)
        return self.decoder((z_class, z_style))

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            z_mean_class, z_log_var_class, z_class, z_mean_style, z_log_var_style, z_style = self.encoder(images)
            reconstruction = self.decoder((z_class, z_style))
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2))
            )
            
            # KL losses
            kl_loss_class = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var_class - tf.square(z_mean_class) - tf.exp(z_log_var_class), axis=1)
            )

            kl_loss_style = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var_style - tf.square(z_mean_style) - tf.exp(z_log_var_style), axis=1)
            )
            kl_loss = (self.beta_class * kl_loss_class) + (self.beta_style * kl_loss_style)
            
            # Classification loss
            class_pred = self.classifier(z_class)
            class_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, class_pred))
            
            # Adversarial loss
            z_style_rev = self.grl(z_style)
            adv_pred = self.adv_classifier(z_style_rev)
            adv_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, adv_pred))
            
            total_loss = reconstruction_loss + kl_loss + class_loss + self.gamma * adv_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
        }

    def test_step(self, data):
        images, labels = data
        z_mean_class, z_log_var_class, z_class, z_mean_style, z_log_var_style, z_style = self.encoder(images)
        reconstruction = self.decoder((z_class, z_style))
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2))
        )
        kl_loss_class = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var_class - tf.square(z_mean_class) - tf.exp(z_log_var_class), axis=1)
        )
        kl_loss_style = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var_style - tf.square(z_mean_style) - tf.exp(z_log_var_style), axis=1)
        )
        kl_loss = (self.beta_class * kl_loss_class) + (self.beta_style * kl_loss_style)
        class_pred = self.classifier(z_class)
        class_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, class_pred))
        z_style_rev = self.grl(z_style)
        adv_pred = self.adv_classifier(z_style_rev)
        adv_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, adv_pred))
        total_loss = reconstruction_loss + kl_loss + class_loss + self.gamma * adv_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim_class": self.latent_dim_class,
            "latent_dim_style": self.latent_dim_style,
            "beta_class": self.beta_class,
            "beta_style": self.beta_style,
            "gamma": self.gamma,
        })
        return config

AAE.Encoder = AAEEncoder
AAE.Decoder = AAEDecoder
AAE.Sampling = Sampling

if __name__ == "__main__":
    import sys
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    latent_dim_class = 8
    latent_dim_style = 8
    beta_class = 6.0
    beta_style = 0.0
    gamma = 10.0
    num_epochs = 20
    batch_size = 32

    vae = AAE(latent_dim_class=latent_dim_class,
              latent_dim_style=latent_dim_style,
              beta_class=beta_class, 
              beta_style=beta_style,
              gamma=gamma)
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    vae.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
            validation_data=(x_test, y_test))
    
    dummy_x = np.random.rand(1, 28, 28, 1).astype("float32")
    _ = vae(dummy_x)

    MODEL_PATH = cache.MODEL_FOLDER / "AAE"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    vae.save(MODEL_PATH / "aae16.keras")