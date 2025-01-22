import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import math

from sklearn.manifold import TSNE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense, Lambda, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow.keras.backend as K # backend différent de keras.backend

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

num_epochs = 6

img_shape = (28, 28, 1)
batch_size = 16

latent_dims = [8, 32, 64, 128]

X_valid_encoded = [None] * len(latent_dims)
X_decoded = [None] * len(latent_dims)

decoder = [None] * len(latent_dims)

for li in range(len(latent_dims)):
    latent_dim = latent_dims[li]

    input_img = Input(shape = img_shape)

    # Encoder
    x = Conv2D(32, 3, padding="same")(input_img)
    x = ReLU()(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = ReLU()(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = ReLU()(x)

    shape_before_flattening = K.int_shape(x)

    x = Flatten()(x)

    z_mu = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(shape = (K.shape(z_mu)[0], latent_dim), mean = 0.0, stddev = 1.0)
        return z_mu + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mu, z_log_sigma])

    # Decoder
    decoder_input = Input(K.int_shape(z)[1:])

    x = Dense(np.prod(shape_before_flattening[1:]))(decoder_input)
    x = ReLU()(x)

    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv2DTranspose(128, 3, padding="same")(x)
    x = ReLU()(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, 3, padding="same")(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, 3, padding="same")(x)
    x = ReLU()(x)

    x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    decoder[li] = Model(decoder_input, x)

    z_decoded = decoder[li](z)

    class CustomVariationalLayer(keras.layers.Layer):
        def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
            kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x, z_decoded, z_mu, z_log_sigma = inputs
            loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
            self.add_loss(loss)
            return x

    y = CustomVariationalLayer()([input_img, z_decoded, z_mu, z_log_sigma])

    vae = Model(input_img, y)
    vae.compile(optimizer = "adam", loss = None)
    vae.summary()

    # Fitting

    vae.fit(x = X_train, y = None, shuffle = True, epochs = num_epochs, batch_size = batch_size, validation_data = (X_valid, None))

    encoder = Model(input_img, z_mu)
    X_valid_encoded[li] = encoder.predict(X_valid, batch_size = batch_size)

    X_decoded[li] = decoder[li].predict(X_valid_encoded[li], batch_size = batch_size)

    encoder.save("./Models/mnist-" + str(latent_dim) + "-encoder.keras")
    decoder[li].save("./Models/mnist-" + str(latent_dim) + "-decoder.keras")

# Évaluation des reconstructions de digits

print("Collecting results...")

digits = [
    [1333, 5526, 972, 5554, 5838],
    [9415, 5889, 672, 831, 7344],
    [3773, 1256, 2082, 3513, 3047],
    [524, 5358, 313, 590, 3085],
    [1980, 4217, 5713, 5200, 3985],
    [1874, 6491, 7583, 3220, 5473],
    [4252, 446, 4466, 7668, 3135],
    [5960, 75, 3309, 9141, 6050],
    [8466, 5729, 7745, 128, 4380],
    [5333, 1088, 1869, 4683, 1005]
]

for i in range(10):
    fig, axes = plt.subplots(5, 1 + len(latent_dims), figsize=(2 * (1 + len(latent_dims)), 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    fig.suptitle("Dimension de l'espace latent", x=0.5, y=0.92, fontsize=18)
    axes[0, 0].set_title("Source", fontsize=14)

    for j in range(len(digits[i])):
        digit_index = digits[i][j]

        axes[j, 0].imshow(X_valid[digit_index].reshape(28, 28))
        axes[j, 0].axis("off")

    for j in range(len(latent_dims)):
        axes[0, 1 + j].set_title(str(latent_dims[j]), fontsize=14)

        for k in range(len(digits[i])):
            digit_index = digits[i][k]
            
            img_src = X_valid[digit_index].reshape(28, 28)
            img_dst = X_decoded[j][digit_index].reshape(28, 28)

            axes[k, 1 + j].imshow(img_dst)

            decoded_psnr = psnr(img_src, img_dst)
            decoded_ssim = ssim(img_src, img_dst, data_range = 1)

            axes[k, 1 + j].axis("off")
            axes[k, 1 + j].text(0.5, -0.4, f"PSNR = {decoded_psnr:.2f} dB\nSSIM = {decoded_ssim:.2f}", fontsize = 12, color = "blue", ha = "center", transform = axes[k, 1 + j].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("./Results/Model-Eval/mnist-vae-eval-" + str(i) + ".png")

# Évaluation des reconstructions moyennes de digits

print("Collecting average results...")

avgs_n = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.002]

for i in range(10):
    fig, axes = plt.subplots(len(avgs_n), len(latent_dims), figsize=(2 * len(latent_dims), 2 * len(avgs_n)))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    fig.suptitle("Dimension de l'espace latent", x=0.5, y=0.92, fontsize=18)
    for j in range(len(avgs_n)):
        prcnt = avgs_n[j] * 100.0
        axes[j, 0].annotate(f"{prcnt:.2f}%", xy=(-0.2, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=12)

    for j in range(len(latent_dims)):
        axes[0, j].set_title(str(latent_dims[j]), fontsize=14)

        for k in range(len(avgs_n)):
            X_digit_encoded = X_valid_encoded[j][Y_valid == i]

            indices = np.random.choice(X_digit_encoded.shape[0], math.ceil(avgs_n[k] * X_digit_encoded.shape[0]), replace=False)

            random_encoded = X_digit_encoded[indices]
        
            mean_encoded = np.mean(random_encoded, axis = 0).reshape(1, -1)

            X_decoded_avg = decoder[j].predict(mean_encoded, batch_size = batch_size, verbose = False)
            axes[k, j].imshow(X_decoded_avg[0])
            axes[k, j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("./Results/Model-Eval/mnist-vae-eval-avg-" + str(i) + ".png")

# Calcul du PSNR et SSIM moyens

print("Calculating mean metrics...")

output = "latent_size,psnr,ssim\n"
for i in range(len(latent_dims)):
    mean_psnr = np.mean([psnr(X_valid[j], X_decoded[i][j]) for j in range(X_valid.shape[0])])
    mean_ssim = np.mean([ssim(X_valid[j], X_decoded[i][j], data_range = 1, channel_axis = -1) for j in range(X_valid.shape[0])])
    output += str(latent_dims[i]) + "," + str(mean_psnr) + "," + str(mean_ssim) + "\n"

with open("./Results/Model-Eval/mnist-vae-eval-metrics.csv", 'w') as f:
    f.write(output)