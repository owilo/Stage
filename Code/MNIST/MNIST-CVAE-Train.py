import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    def __init__(self, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
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

class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate()
        self.dense0 = layers.Dense(256, activation='relu')
        self.dense = layers.Dense(7*7*64, activation='relu')
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

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype("float32") / 255.
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

latent_dim = 24
cvae = CVAE(latent_dim=latent_dim, final_beta=3.0, annealing_steps=20000)
cvae.compile(optimizer=keras.optimizers.Adam())

cvae.fit(
    x=x_train, y=y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test)
)

digits = [
    1333,  # 0
    9415,  # 1
    3773,  # 2
    524,   # 3
    1980,  # 4
    1874,  # 5
    4252,  # 6
    6960,  # 7
    8466,  # 8
    5333   # 9
]

fig, axs = plt.subplots(10, 11, figsize=(11, 10))

for j, idx in enumerate(digits):
    x_src = x_test[idx:idx + 1]
    y_src = y_test[idx:idx + 1]
    
    _, _, z = cvae.encoder(x_src)
    
    axs[j, 0].imshow(x_src[0, :, :, 0], cmap='gray')
    axs[j, 0].axis('off')
    
    for i in range(10):
        y_target = np.zeros((1, num_classes), dtype="float32")
        y_target[0, i] = 1.0
        
        x_decoded = cvae.decoder((z, y_target))
        
        axs[j, i + 1].imshow(x_decoded[0, :, :, 0], cmap='gray')
        axs[j, i + 1].axis('off')

plt.suptitle("CVAE", fontsize=16)
plt.tight_layout()
plt.savefig("./Results/mnist-cvae.png")

z_mean, _, _ = cvae.encoder(x_test)
z_mean = z_mean.numpy()

tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z_mean)

labels = np.argmax(y_test, axis=1)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="Paired", alpha=0.35)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE CVAE (style)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./Results/mnist-cvae-style-tsne.png")

z_mean_class = np.concatenate((z_mean, y_test.astype(float)), axis=1)

tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z_mean_class)

labels = np.argmax(y_test, axis=1)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="Paired", alpha=0.35)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE CVAE")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./Results/mnist-cvae-tsne.png")