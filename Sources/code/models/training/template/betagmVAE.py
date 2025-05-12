import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from code.models.common.layers import Sampling

@tf.keras.utils.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, latent_dim=2, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()
        self.classifier = layers.Dense(num_classes, activation='softmax', name='classifier')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        z = self.sampling((mean, log_var))
        logits = self.classifier(z)
        return mean, log_var, z, logits

    def get_config(self):
        return super().get_config()

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
class SupervisedGMMVAE(keras.Model):
    def __init__(self, latent_dim=2, num_classes=10, beta=1.0, low_rank_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta
        self.low_rank_dim = low_rank_dim
        self.encoder = Encoder(latent_dim=latent_dim, num_classes=num_classes)
        self.decoder = Decoder()
        # Prior parameters
        self.prior_mu = tf.Variable(tf.random.normal([num_classes, latent_dim]), name='prior_mu')
        self.prior_log_diag_var = tf.Variable(tf.zeros([num_classes, latent_dim]), name='prior_log_diag_var')
        self.prior_U = tf.Variable(tf.random.normal([num_classes, latent_dim, low_rank_dim]) * 0.01, name='prior_U')
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.rec_loss_tracker = keras.metrics.Mean(name='rec_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.class_loss_tracker = keras.metrics.Mean(name='class_loss')
        self.class_accuracy = keras.metrics.SparseCategoricalAccuracy(name='class_acc')

    def compute_kl_full_cov(self, mean, log_var, y):
        # Gather per-class prior parameters
        y = tf.cast(y, dtype=tf.int32)
        mu_k = tf.gather(self.prior_mu, y)                        # [B, D]
        log_diag_var_k = tf.gather(self.prior_log_diag_var, y)   # [B, D]
        U_k = tf.gather(self.prior_U, y)                          # [B, D, r]

        # Covariance: Σ_k = diag + UUᵀ
        diag_k = tf.exp(log_diag_var_k)                           # [B, D]
        diag_inv = tf.math.reciprocal(diag_k)                     # [B, D]
        diff = mean - mu_k                                        # [B, D]

        # Compute Mahalanobis distance: (z - μ)ᵀ Σ⁻¹ (z - μ)
        # Efficient solve using Woodbury identity
        diag_inv_mat = tf.linalg.diag(diag_inv)                   # [B, D, D]
        U_transpose = tf.transpose(U_k, [0, 2, 1])                # [B, r, D]
        A = tf.linalg.matmul(U_transpose, diag_inv_mat)          # [B, r, D]
        B = tf.linalg.matmul(A, U_k)                              # [B, r, r]
        I = tf.eye(self.low_rank_dim, batch_shape=[tf.shape(B)[0]])
        B_inv = tf.linalg.inv(B + I)                              # [B, r, r]
        quad_term = tf.reduce_sum(diag_inv * diff**2, axis=1, keepdims=True) \
                  - tf.reduce_sum(tf.linalg.matmul(A, tf.expand_dims(diff, -1))**2, axis=(1, 2), keepdims=True) \
                  + tf.reduce_sum(tf.linalg.matmul(B_inv, tf.linalg.matmul(A, tf.expand_dims(diff, -1)))**2, axis=(1, 2), keepdims=True)

        # Compute log|Σ_k|
        log_det_diag = tf.reduce_sum(log_diag_var_k, axis=1, keepdims=True)   # [B,1]
        log_det_low_rank = tf.linalg.logdet(I + B)                            # [B]
        log_det_total = log_det_diag + tf.expand_dims(log_det_low_rank, -1)

        trace_term = tf.reduce_sum(tf.exp(log_var) * diag_inv, axis=1, keepdims=True)

        kl = 0.5 * (log_det_total - tf.reduce_sum(log_var, axis=1, keepdims=True) +
                    trace_term + quad_term - self.latent_dim)
        return tf.reduce_mean(kl)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            mean, log_var, z, logits = self.encoder(x, training=True)
            x_recon = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2))
            )
            kl_loss = self.compute_kl_full_cov(mean, log_var, y)
            class_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y, logits)
            )
            total_loss = recon_loss + self.beta * kl_loss + class_loss
        grads = tape.gradient(total_loss, self.trainable_weights + [self.prior_mu, self.prior_log_diag_var, self.prior_U])
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights + [self.prior_mu, self.prior_log_diag_var, self.prior_U]))

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        mean, log_var, z, logits = self.encoder(x, training=False)
        x_recon = self.decoder(z)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2))
        )
        kl_loss = self.compute_kl_full_cov(mean, log_var, y)
        class_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, logits)
        )
        total_loss = recon_loss + self.beta * kl_loss + class_loss
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker,
            self.class_accuracy
        ]

    def call(self, inputs, y=None):
        mean, log_var, z, _ = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        config = super().get_config()
        config.update({'latent_dim': self.latent_dim,
                       'num_classes': self.num_classes,
                       'beta': self.beta})
        return config


if __name__ == '__main__':
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent_dim', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-c', '--num_classes', type=int, default=10)
    parser.add_argument('--beta', type=float, default=4.0)
    args = parser.parse_args()

    model = SupervisedGMMVAE(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        beta=args.beta
    )
    model.compile(optimizer=keras.optimizers.Adam())

    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test)
    )

    # --- Testing / Visualization ---
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 1) t-SNE of latent space
    mean, log_var, z, _ = model.encoder(x_test, training=False)
    z_np = z.numpy()
    tsne = TSNE(n_components=2, random_state=42)
    z_embedded = tsne.fit_transform(z_np)
    plt.figure(figsize=(8, 8))
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=y_test, cmap='tab10', s=5)
    plt.colorbar()
    plt.title('t-SNE of latent representations')
    plt.show()

    # 2) Sample reconstructions for each class
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        idxs = np.where(y_test == digit)[0][:10]
        _, _, z_sel, _ = model.encoder(x_test[idxs], training=False)
        recon = model.decoder(z_sel)
        for i, ax in enumerate(axs[digit]):
            ax.imshow(recon[i].numpy().squeeze(), cmap='gray')
            ax.axis('off')
    plt.suptitle('Reconstructions: 10 samples per class')
    plt.show()

    # 3) Latent space translations with full affine transform
    from scipy.linalg import sqrtm, inv

    fig, axs = plt.subplots(10, 10, figsize=(10, 10))

    for src in range(10):
        x_src = x_test[y_test == src][0:1]
        mean, log_var, z, _ = model.encoder(x_src, training=False)

        mu_src = model.prior_mu[src].numpy()
        diag_src = np.exp(model.prior_log_diag_var[src].numpy())
        U_src = model.prior_U[src].numpy()
        cov_src = np.diag(diag_src) + U_src @ U_src.T

        for tgt in range(10):
            mu_tgt = model.prior_mu[tgt].numpy()
            diag_tgt = np.exp(model.prior_log_diag_var[tgt].numpy())
            U_tgt = model.prior_U[tgt].numpy()
            cov_tgt = np.diag(diag_tgt) + U_tgt @ U_tgt.T

            # Compute transformation: A = sqrt(Sigma_tgt) @ inv_sqrt(Sigma_src)
            sqrt_cov_src = sqrtm(cov_src)
            inv_sqrt_cov_src = inv(sqrt_cov_src)
            sqrt_cov_tgt = sqrtm(cov_tgt)
            A = sqrt_cov_tgt @ inv_sqrt_cov_src

            z_np = z.numpy().squeeze()
            z_transformed = A @ (z_np - mu_src) + mu_tgt
            z_transformed = tf.convert_to_tensor([z_transformed], dtype=tf.float32)

            # Decode and plot
            img = model.decoder(z_transformed)[0]
            axs[src, tgt].imshow(img.numpy().squeeze(), cmap='gray')
            axs[src, tgt].axis('off')

    plt.suptitle('Latent Space Affine Transforms Between Class Gaussians')
    plt.tight_layout()
    plt.show()