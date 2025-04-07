import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Sampling layer used by both branches
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Gradient reversal layer for adversarial loss on the style branch
@tf.keras.utils.register_keras_serializable()
class GradientReversal(layers.Layer):
    def call(self, x):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -dy  # reverses the gradient
            return x, grad
        return reverse_gradient(x)

# Dual Encoder: produces two latent vectors: one for class and one for style.
@tf.keras.utils.register_keras_serializable()
class DualEncoder(keras.Model):
    def __init__(self, class_dim=2, style_dim=2, num_classes=10, **kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.class_dim = class_dim
        self.style_dim = style_dim
        self.num_classes = num_classes
        # Shared convolutional layers
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        # Class branch: produce mean, logvar and sample z_class; also add a classifier
        self.class_z_mean = layers.Dense(class_dim)
        self.class_z_log_var = layers.Dense(class_dim)
        self.class_sampling = Sampling()
        self.classifier = layers.Dense(num_classes, activation='softmax')
        # Style branch: produce mean, logvar and sample z_style
        self.style_z_mean = layers.Dense(style_dim)
        self.style_z_log_var = layers.Dense(style_dim)
        self.style_sampling = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        # Class branch
        class_z_mean = self.class_z_mean(x)
        class_z_log_var = self.class_z_log_var(x)
        z_class = self.class_sampling((class_z_mean, class_z_log_var))
        class_pred = self.classifier(z_class)
        # Style branch
        style_z_mean = self.style_z_mean(x)
        style_z_log_var = self.style_z_log_var(x)
        z_style = self.style_sampling((style_z_mean, style_z_log_var))
        return (class_z_mean, class_z_log_var, z_class,
                style_z_mean, style_z_log_var, z_style, class_pred)
    
    def get_config(self):
        config = super(DualEncoder, self).get_config()
        config.update({
            "class_dim": self.class_dim,
            "style_dim": self.style_dim,
            "num_classes": self.num_classes,
        })
        return config

# Dual Decoder: concatenates class and style latent vectors and decodes them.
@tf.keras.utils.register_keras_serializable()
class DualDecoder(keras.Model):
    def __init__(self, **kwargs):
        super(DualDecoder, self).__init__(**kwargs)
        self.concat = layers.Concatenate()
        self.dense0 = layers.Dense(256, activation='relu')
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        # inputs is a tuple: (z_class, z_style)
        z_class, z_style = inputs
        x = self.concat([z_class, z_style])
        x = self.dense0(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.conv_out(x)
    
    def get_config(self):
        config = super(DualDecoder, self).get_config()
        return config

# Style adversary: takes the style latent and tries to predict the class.
# With the gradient reversal layer, the encoder is trained to remove class info from z_style.
@tf.keras.utils.register_keras_serializable()
class StyleAdversary(keras.Model):
    def __init__(self, num_classes=10, **kwargs):
        super(StyleAdversary, self).__init__(**kwargs)
        self.grl = GradientReversal()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        x = self.grl(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def get_config(self):
        config = super(StyleAdversary, self).get_config()
        return config

# The DualCVAE model that brings everything together.
@tf.keras.utils.register_keras_serializable()
class DualCVAE(keras.Model):
    def __init__(self, class_dim=2, style_dim=2, num_classes=10,
                 final_beta=4.0, annealing_steps=6000,
                 cls_loss_weight=1.0, adv_loss_weight=1.0, **kwargs):
        super(DualCVAE, self).__init__(**kwargs)
        self.class_dim = class_dim
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.final_beta = final_beta
        self.annealing_steps = annealing_steps
        self.cls_loss_weight = cls_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.encoder = DualEncoder(class_dim=class_dim, style_dim=style_dim, num_classes=num_classes)
        self.decoder = DualDecoder()
        self.style_adversary = StyleAdversary(num_classes=num_classes)
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker,
            self.adv_loss_tracker,
        ]
    
    def compute_kl_loss(self, z_mean, z_log_var):
        kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        return tf.reduce_mean(kl)
    
    def train_step(self, data):
        # data is a tuple (x, y) with y as one-hot labels
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            # Encode: get both latent spaces and the classifier output from the class branch.
            (class_z_mean, class_z_log_var, z_class,
             style_z_mean, style_z_log_var, z_style, class_pred) = self.encoder(x)
            # Decode: reconstruct the input image from the concatenated latent vectors.
            x_recon = self.decoder((z_class, z_style))
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
                )
            )
            # KL divergence for both latent spaces
            kl_loss_class = self.compute_kl_loss(class_z_mean, class_z_log_var)
            kl_loss_style = self.compute_kl_loss(style_z_mean, style_z_log_var)
            step = tf.cast(self.optimizer.iterations, tf.float32)
            beta = self.final_beta * tf.minimum(1.0, step / self.annealing_steps)
            kl_loss = beta * (kl_loss_class + kl_loss_style)
            # Classification loss on the class branch (supervising z_class)
            cls_loss = keras.losses.categorical_crossentropy(y, class_pred)
            cls_loss = tf.reduce_mean(cls_loss)
            # Adversarial loss: the style adversary attempts to predict the class from z_style.
            adv_pred = self.style_adversary(z_style)
            adv_loss = keras.losses.categorical_crossentropy(y, adv_pred)
            adv_loss = tf.reduce_mean(adv_loss)
            # Total loss: combine all loss components.
            total_loss = reconstruction_loss + kl_loss + self.cls_loss_weight * cls_loss + self.adv_loss_weight * adv_loss
        # Compute gradients and update weights.
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metrics.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(cls_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
            "beta": beta,
        }
    
    def test_step(self, data):
        x, y = data
        (class_z_mean, class_z_log_var, z_class,
         style_z_mean, style_z_log_var, z_style, class_pred) = self.encoder(x)
        x_recon = self.decoder((z_class, z_style))
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2)
            )
        )
        kl_loss_class = self.compute_kl_loss(class_z_mean, class_z_log_var)
        kl_loss_style = self.compute_kl_loss(style_z_mean, style_z_log_var)
        beta = self.final_beta
        kl_loss = beta * (kl_loss_class + kl_loss_style)
        cls_loss = keras.losses.categorical_crossentropy(y, class_pred)
        cls_loss = tf.reduce_mean(cls_loss)
        adv_pred = self.style_adversary(z_style)
        adv_loss = keras.losses.categorical_crossentropy(y, adv_pred)
        adv_loss = tf.reduce_mean(adv_loss)
        total_loss = reconstruction_loss + kl_loss + self.cls_loss_weight * cls_loss + self.adv_loss_weight * adv_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(cls_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
        }
    
    def call(self, inputs):
        x, _ = inputs
        (class_z_mean, class_z_log_var, z_class,
         style_z_mean, style_z_log_var, z_style, _) = self.encoder(x)
        return self.decoder((z_class, z_style))
    
    def get_config(self):
        config = super(DualCVAE, self).get_config()
        config.update({
            "class_dim": self.class_dim,
            "style_dim": self.style_dim,
            "num_classes": self.num_classes,
            "final_beta": self.final_beta,
            "annealing_steps": self.annealing_steps,
            "cls_loss_weight": self.cls_loss_weight,
            "adv_loss_weight": self.adv_loss_weight,
        })
        return config

# ================================
# Data preparation (using MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Instantiate and compile the model.
dual_cvae = DualCVAE(class_dim=16, style_dim=2, num_classes=10,
                     final_beta=4.0, annealing_steps=6000,
                     cls_loss_weight=1.0, adv_loss_weight=1.0)
dual_cvae.compile(optimizer=keras.optimizers.Adam())

# Train the model.
dual_cvae.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))

# ================================
# Reconstruction and Style Transfer Tests

# Function to plot original images and their reconstructions.
def plot_reconstructions(model, images, num_images=10):
    (class_z_mean, class_z_log_var, z_class,
     style_z_mean, style_z_log_var, z_style, class_pred) = model.encoder(images)
    x_recon = model.decoder((z_class, z_style))
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Original image.
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")
        # Reconstructed image.
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(x_recon[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()

# Function to conduct style transfer: keep class latent fixed, vary style latent.
def plot_style_transfer_bilinear(model, image, grid_size=10, style_range=(-2, 2)):
    """
    Generates a bilinear grid plot over the style latent space while keeping
    the class latent fixed. The style latent is assumed to be 2D.
    
    Args:
      model: The trained DualCVAE model.
      image: A single input image.
      grid_size: Number of steps in each dimension of the grid.
      style_range: Tuple (min, max) for the range of each style latent dimension.
    """
    # Encode a single image to get a fixed class latent vector.
    (class_z_mean, class_z_log_var, z_class,
     style_z_mean, style_z_log_var, z_style, class_pred) = model.encoder(tf.expand_dims(image, 0))
    
    # Generate a grid in the style latent space.
    grid_x = np.linspace(style_range[0], style_range[1], grid_size)
    grid_y = np.linspace(style_range[0], style_range[1], grid_size)
    style_grid = []
    for yi in grid_y:
        for xi in grid_x:
            style_grid.append([xi, yi])
    style_grid = np.array(style_grid, dtype=np.float32)
    
    # Repeat the fixed class latent vector for each point in the grid.
    z_class_rep = tf.repeat(z_class, repeats=grid_size * grid_size, axis=0)
    
    # Decode the grid of latent vectors.
    x_decoded = model.decoder((z_class_rep, style_grid))
    
    # Assemble the decoded images into a grid for display.
    # Here we assume MNIST images of size 28x28.
    digit_size = x_decoded.shape[1]
    figure = np.zeros((digit_size * grid_size, digit_size * grid_size))
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            digit = x_decoded[i * grid_size + j].numpy().squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.show()

# Run reconstruction test on a batch from the test set.
sample_images = x_test[:10]
plot_reconstructions(dual_cvae, sample_images)

# Run a style transfer test on a single test image.
plot_style_transfer_bilinear(dual_cvae, x_test[0])