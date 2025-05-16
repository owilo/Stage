import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import argparse

from code.models.common.layers import Sampling

@tf.keras.utils.register_keras_serializable()
class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 mask_type='B',
                 padding='same',
                 activation=None,
                 use_bias=True,
                 strides=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type.upper()
        assert self.mask_type in {'A', 'B'}
        self.padding = padding
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        # Create the convolution layer
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=self.use_bias,
            strides=self.strides,
            activation=None
        )

    def build(self, input_shape):
        # Build the convolution layer
        self.conv.build(input_shape)

        # Create the mask as a trainable weight
        kh, kw, in_ch, out_ch = self.conv.kernel.shape
        mask = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
        center_h, center_w = kh // 2, kw // 2

        # Zero out the mask for rows below and including the center based on mask_type
        mask[center_h + 1:, :, :, :] = 0
        # Corrected line: for 'A', block center_w onwards; for 'B', block center_w +1 onwards
        mask[center_h, center_w + (self.mask_type == 'B'):, :, :] = 0

        # Add mask as a constant weight
        self.mask = self.add_weight(
            name="mask",
            shape=(kh, kw, in_ch, out_ch),
            initializer=tf.keras.initializers.Constant(mask),
            trainable=False
        )

        super().build(input_shape)

    def call(self, inputs):
        # Apply the mask to the kernel weights
        masked_kernel = self.conv.kernel * self.mask

        # Perform convolution with masked kernel
        x = tf.nn.conv2d(
            inputs,
            masked_kernel,
            strides=(1, *self.strides, 1),
            padding=self.padding.upper()
        )

        # Add bias if needed
        if self.use_bias:
            x = tf.nn.bias_add(x, self.conv.bias)

        # Apply activation function if specified
        if self.activation is not None:
            x = self.activation(x)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'mask_type': self.mask_type,
            'padding': self.padding,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'strides': self.strides
        })
        return cfg
    
@tf.keras.utils.register_keras_serializable()
class GatedMaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 mask_type='B',
                 padding='same',
                 use_bias=True,
                 strides=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters      = filters
        self.kernel_size  = kernel_size
        self.mask_type    = mask_type.upper()
        assert self.mask_type in {'A', 'B'}
        self.padding      = padding
        self.strides      = (strides, strides) if isinstance(strides, int) else strides
        self.use_bias     = use_bias

        # Two parallel convs: one for tanh branch, one for sigmoid branch
        self.conv_tanh   = layers.Conv2D(filters, kernel_size,
                                         padding=padding,
                                         strides=self.strides,
                                         use_bias=use_bias,
                                         activation=None)
        self.conv_sigmoid = layers.Conv2D(filters, kernel_size,
                                          padding=padding,
                                          strides=self.strides,
                                          use_bias=use_bias,
                                          activation=None)

    def build(self, input_shape):
        # Build both convs to get their kernels
        self.conv_tanh.build(input_shape)
        self.conv_sigmoid.build(input_shape)

        # Create shared mask tensor
        kh, kw, in_ch, out_ch = self.conv_tanh.kernel.shape
        mask = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
        center_h, center_w = kh // 2, kw // 2
        mask[center_h + 1:, :, :, :] = 0
        mask[center_h, center_w + (self.mask_type == 'B'):, :, :] = 0

        # Register as non‑trainable weight
        self.mask = self.add_weight(
            name="mask",
            shape=(kh, kw, in_ch, out_ch),
            initializer=tf.keras.initializers.Constant(mask),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        # Apply mask to both conv kernels
        kt = self.conv_tanh.kernel * self.mask
        ks = self.conv_sigmoid.kernel * self.mask

        # Manual conv2d so we can use masked kernels
        x_t = tf.nn.conv2d(inputs, kt, strides=(1, *self.strides, 1),
                           padding=self.padding.upper())
        x_s = tf.nn.conv2d(inputs, ks, strides=(1, *self.strides, 1),
                           padding=self.padding.upper())

        if self.use_bias:
            x_t = tf.nn.bias_add(x_t, self.conv_tanh.bias)
            x_s = tf.nn.bias_add(x_s, self.conv_sigmoid.bias)

        # Gated activation
        return tf.math.tanh(x_t) * tf.math.sigmoid(x_s)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'filters':    self.filters,
            'kernel_size':self.kernel_size,
            'mask_type':  self.mask_type,
            'padding':    self.padding,
            'use_bias':   self.use_bias,
            'strides':    self.strides,
        })
        return cfg

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
    
class SimpleDecoder(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        # small deconv backbone for quick global recon
        self.dense = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.conv_t1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.conv_t2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.out = layers.Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        return self.out(x)

# ---------------- Shallow PixelCNN++ Refiner (Stage 2) ----------------
@tf.keras.utils.register_keras_serializable()
class ShallowPixelCNNDecoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, hidden_channels=64, n_maskB_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(hidden_channels, 1, padding='same')
        self.merge_proj = layers.Conv2D(hidden_channels, 1, padding='same')
        self.maskA = GatedMaskedConv2D(hidden_channels, 7, mask_type='A', padding='same')

        # Dynamically create n MaskedConv2D layers for mask type B
        self.maskB_layers = [
            GatedMaskedConv2D(hidden_channels, 3, mask_type='B', padding='same')
            for _ in range(n_maskB_layers)
        ]

        self.out_conv = MaskedConv2D(1, 1, mask_type='B', activation='sigmoid')

    def call(self, x, z=None):
        h = self.proj(x)  # [B, 28, 28, H]

        if z is not None:
            z_broadcast = tf.expand_dims(tf.expand_dims(z, 1), 1)
            z_broadcast = tf.tile(z_broadcast, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
            h = tf.concat([h, z_broadcast], axis=-1)
            h = self.merge_proj(h)

        h1 = self.maskA(h)

        # Initialize residual as the first masked layer (h1 + h)
        residual = h1 + h
        
        # Apply the residuals using a for loop for mask B layers
        for maskB_layer in self.maskB_layers:
            residual = maskB_layer(tf.nn.relu(residual))  # Apply each MaskedConv2D layer
        
        return self.out_conv(tf.nn.relu(residual + h1 + h))

import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
class SupervisedGMMVAE(keras.Model):
    def __init__(
        self,
        latent_dim=2,
        num_classes=10,
        beta=1.0,
        low_rank_dim=2,
        lambda_ssim=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = tf.Variable(0.0, trainable=False)
        self.low_rank_dim = low_rank_dim
        self.lambda_ssim = lambda_ssim

        # Encoder / Decoder
        self.encoder = Encoder(latent_dim=latent_dim, num_classes=num_classes)
        self.decoder = SimpleDecoder(latent_dim)

        # Prior parameters
        self.prior_mu = tf.Variable(
            tf.random.normal([num_classes, latent_dim]), name='prior_mu'
        )
        self.prior_log_diag_var = tf.Variable(
            tf.zeros([num_classes, latent_dim]), name='prior_log_diag_var'
        )
        self.prior_U = tf.Variable(
            tf.random.normal([num_classes, latent_dim, low_rank_dim]) * 0.01,
            name='prior_U'
        )

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.rec_bce_tracker   = keras.metrics.Mean(name='bce_loss')
        self.ssim_loss_tracker = keras.metrics.Mean(name='ssim_loss')
        self.kl_loss_tracker   = keras.metrics.Mean(name='kl_loss')
        self.class_loss_tracker= keras.metrics.Mean(name='class_loss')
        self.class_accuracy    = keras.metrics.SparseCategoricalAccuracy(name='class_acc')

    def compute_kl_full_cov(self, mean, log_var, y):
        y = tf.cast(y, tf.int32)
        mu_k          = tf.gather(self.prior_mu, y)
        log_var_k     = tf.gather(self.prior_log_diag_var, y)
        U_k           = tf.gather(self.prior_U, y)
        diag_k        = tf.exp(log_var_k)
        diag_inv      = tf.math.reciprocal(diag_k)
        diff          = mean - mu_k

        # Woodbury‐based quadratic term
        diag_inv_mat = tf.linalg.diag(diag_inv)
        U_T          = tf.transpose(U_k, [0,2,1])
        A            = U_T @ diag_inv_mat
        B            = A @ U_k
        I_r          = tf.eye(self.low_rank_dim, batch_shape=[tf.shape(B)[0]])
        B_inv        = tf.linalg.inv(B + I_r)

        quad_term = (
            tf.reduce_sum(diag_inv * diff**2, axis=1, keepdims=True)
            - tf.reduce_sum((A @ tf.expand_dims(diff, -1))**2, axis=(1,2), keepdims=True)
            + tf.reduce_sum((B_inv @ (A @ tf.expand_dims(diff, -1)))**2, axis=(1,2), keepdims=True)
        )

        # Log‐determinant
        log_det_diag    = tf.reduce_sum(log_var_k, axis=1, keepdims=True)
        log_det_lowrank = tf.linalg.logdet(I_r + B)
        log_det_total   = log_det_diag + tf.expand_dims(log_det_lowrank, -1)

        trace_term = tf.reduce_sum(tf.exp(log_var) * diag_inv, axis=1, keepdims=True)

        kl = 0.5 * (
            log_det_total
            - tf.reduce_sum(log_var, axis=1, keepdims=True)
            + trace_term
            + quad_term
            - self.latent_dim
        )
        return tf.reduce_mean(kl)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            mean, log_var, z, logits = self.encoder(x, training=True)
            x_recon = self.decoder(z)

            # BCE reconstruction
            bce = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon),
                    axis=(1,2)
                )
            )

            # SSIM reconstruction
            ssim_vals = tf.image.ssim(x, x_recon, max_val=1.0)   # [B]
            ssim_loss = tf.reduce_mean(1.0 - ssim_vals)

            # combined recon loss
            recon_loss = bce + self.lambda_ssim * ssim_loss

            kl_loss    = self.compute_kl_full_cov(mean, log_var, y)
            class_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y, logits)
            )

            total_loss = recon_loss + self.beta * kl_loss + class_loss

        grads = tape.gradient(
            total_loss,
            self.trainable_weights + [self.prior_mu, self.prior_log_diag_var, self.prior_U]
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.trainable_weights + [self.prior_mu, self.prior_log_diag_var, self.prior_U]
            )
        )

        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.rec_bce_tracker.update_state(bce)
        self.ssim_loss_tracker.update_state(ssim_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        mean, log_var, z, logits = self.encoder(x, training=False)
        x_recon = self.decoder(z)

        bce = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon),
                axis=(1,2)
            )
        )
        ssim_vals = tf.image.ssim(x, x_recon, max_val=1.0)
        ssim_loss = tf.reduce_mean(1.0 - ssim_vals)
        recon_loss = bce + self.lambda_ssim * ssim_loss

        kl_loss    = self.compute_kl_full_cov(mean, log_var, y)
        class_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, logits)
        )
        total_loss = recon_loss + self.beta * kl_loss + class_loss

        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.rec_bce_tracker.update_state(bce)
        self.ssim_loss_tracker.update_state(ssim_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y, logits)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_bce_tracker,
            self.ssim_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker,
            self.class_accuracy
        ]

    def call(self, inputs, y=None):
        mean, log_var, z, _ = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'beta': float(self.beta.numpy()),
            'low_rank_dim': self.low_rank_dim,
            'lambda_ssim': self.lambda_ssim
        })
        return config

# ---------------- Beta Scheduling Callback ----------------
class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, target_beta, ramp_epochs):
        super().__init__()
        self.target = target_beta
        self.ramp = ramp_epochs
    def on_epoch_begin(self, epoch, logs=None):
        # linear ramp
        beta = (self.target * min(epoch, self.ramp) / self.ramp)
        # assume model.beta is tf.Variable
        self.model.beta.assign(beta)

if __name__ == '__main__':
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent_dim', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=80)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-c', '--num_classes', type=int, default=10)
    parser.add_argument('--beta', type=float, default=4.0)
    args = parser.parse_args()

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    stage1 = SupervisedGMMVAE(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        beta=args.beta,
        lambda_ssim=2.0,
    )
    stage1.compile(optimizer=keras.optimizers.Adam())
    stage1.fit(
        x_train, y_train,
        epochs=int(args.epochs),
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[BetaScheduler(target_beta=args.beta, ramp_epochs=5), lr_scheduler]
    )

    def make_refine_dataset(x_data):
        ds = tf.data.Dataset.from_tensor_slices(x_data)
        ds = ds.batch(args.batch_size)
        # map each batch to ((coarse_images, latents), original_images)
        ds = ds.map(
            lambda batch_x: (
                (
                    stage1.decoder(stage1.encoder(batch_x, training=False)[2]),  # coarse
                    stage1.encoder(batch_x, training=False)[2]                     # z
                ),
                batch_x                                                         # target
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_refine_ds = make_refine_dataset(x_train)
    val_refine_ds = make_refine_dataset(x_test)

    refiner = ShallowPixelCNNDecoder(latent_dim=args.latent_dim)
    inp_coarse = keras.Input(shape=(28,28,1), name='coarse_input')
    inp_z = keras.Input(shape=(args.latent_dim,), name='z_input')
    out_refined = refiner(inp_coarse, inp_z)
    refine_model = keras.Model([inp_coarse, inp_z], out_refined, name='refiner')
    refine_model.compile(optimizer=keras.optimizers.Adam(),
                         loss=keras.losses.binary_crossentropy)

    refine_model.fit(
        train_refine_ds,
        epochs=int(1),
        validation_data=val_refine_ds
    )

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.manifold import TSNE
    from scipy.linalg import sqrtm, inv

    # --- 1) t-SNE of latent space before & after refinement (with batching) ---

    # encode once
    _, _, z, _ = stage1.encoder(x_test, training=False)
    z_np = z.numpy()

    # t-SNE on original z
    tsne = TSNE(n_components=2, random_state=42)
    z_emb = tsne.fit_transform(z_np)
    plt.figure(figsize=(8,8))
    plt.scatter(z_emb[:,0], z_emb[:,1], c=y_test, cmap='tab10', s=5)
    plt.colorbar()
    plt.title('t-SNE of latent codes (before refiner)')
    plt.savefig('fig1.png', dpi=150)
    plt.clf()

    # helper to get refined z in batches
    def get_refined_z(z_tensor, batch_size=1000):
        zs = []
        n = z_tensor.shape[0]
        for i in range(0, n, batch_size):
            z_batch = z_tensor[i:i+batch_size]
            coarse = stage1.decoder(z_batch)
            refined = refiner(coarse, z_batch)
            _, _, z2, _ = stage1.encoder(refined, training=False)
            zs.append(z2.numpy())
        return np.vstack(zs)

    # run batching
    z_refined_np = get_refined_z(z, batch_size=1000)

    # t-SNE on refined z
    zr_emb = TSNE(n_components=2, random_state=42).fit_transform(z_refined_np)
    plt.figure(figsize=(8,8))
    plt.scatter(zr_emb[:,0], zr_emb[:,1], c=y_test, cmap='tab10', s=5)
    plt.colorbar()
    plt.title('t-SNE of latent codes (after refiner)')
    plt.savefig('fig1refined.png', dpi=150)
    plt.clf()


    # --- 2) Sample reconstructions per class before & after refinement ---

    # Coarse reconstructions
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        idxs = np.where(y_test == digit)[0][:10]
        _, _, z_sel, _ = stage1.encoder(x_test[idxs], training=False)
        coarse_sel = stage1.decoder(z_sel)
        for i, ax in enumerate(axs[digit]):
            ax.imshow(coarse_sel[i].numpy().squeeze(), cmap='gray')
            ax.axis('off')
    plt.suptitle('Coarse reconstructions: 10 samples per class')
    plt.savefig('fig2.png', dpi=150)
    plt.clf()

    # Refined reconstructions
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        idxs = np.where(y_test == digit)[0][:10]
        _, _, z_sel, _ = stage1.encoder(x_test[idxs], training=False)
        coarse_sel = stage1.decoder(z_sel)
        refined_sel = refiner(coarse_sel, z_sel)
        for i, ax in enumerate(axs[digit]):
            ax.imshow(refined_sel[i].numpy().squeeze(), cmap='gray')
            ax.axis('off')
    plt.suptitle('Refined reconstructions: 10 samples per class')
    plt.savefig('fig2refined.png', dpi=150)
    plt.clf()


    # --- 3) Latent-space affine transforms before & after refinement ---

    # Precompute per-class covariances
    priors = []
    for c in range(10):
        mu = stage1.prior_mu[c].numpy()
        diag = np.exp(stage1.prior_log_diag_var[c].numpy())
        U = stage1.prior_U[c].numpy()
        cov = np.diag(diag) + U @ U.T
        priors.append((mu, cov))

    # Helper to do one grid
    def make_grid(save_name, refine=False):
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for src in range(10):
            x_src = x_test[y_test == src][:1]
            mean, log_var, z, _ = stage1.encoder(x_src, training=False)
            mu_src, cov_src = priors[src]
            sqrt_cov_src = sqrtm(cov_src)
            inv_sqrt_cov_src = inv(sqrt_cov_src)
            z_np = z.numpy().squeeze()
            for tgt in range(10):
                mu_tgt, cov_tgt = priors[tgt]
                sqrt_cov_tgt = sqrtm(cov_tgt)
                A = sqrt_cov_tgt @ inv_sqrt_cov_src
                z_t = A @ (z_np - mu_src) + mu_tgt
                z_t = tf.convert_to_tensor([z_t], dtype=tf.float32)
                coarse_img = stage1.decoder(z_t)
                img = refiner(coarse_img, z_t) if refine else coarse_img
                axs[src, tgt].imshow(img[0].numpy().squeeze(), cmap='gray')
                axs[src, tgt].axis('off')
        title = ('Refined' if refine else 'Coarse') + ' Affine Transforms Between Class Gaussians'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)
        plt.clf()

    make_grid('fig3.png', refine=False)
    make_grid('fig3refined.png', refine=True)