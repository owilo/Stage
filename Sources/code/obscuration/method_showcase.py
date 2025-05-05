import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import metrics

from code.utils import cache, latent, utils, models, obscuration

utils.deterministic()
utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

betavae, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "category": "BetaVAE", "dataset_range": (0, 1)}
))

cvae, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "category": "CVAE", "dataset_range": (0, 1)}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))

digit = 1980
x_src = x_test[digit: digit + 1]
y_src = y_test[digit: digit + 1]

z_src_betavae = latent.encode(
    betavae,
    x=x_src,
    y=y_src,
    n_times=2,
    save_cache=False
)

z_src_cvae = latent.encode(
    cvae,
    x=x_src,
    y=y_src,
    n_times=2,
    save_cache=False,
    num_classes=10
)

z_class_distributions = latent.encode_class_distributions(
    betavae,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

plt.imsave(
    cache.RESULTS_FOLDER / "ObscurationMethods" / "obs_original.png", 
    x_src[0].squeeze(), cmap='gray'
)

results_file = cache.RESULTS_FOLDER / "ObscurationMethods" / "metrics.txt"
with open(results_file, 'w') as f:
    pass

def save_with_metrics(image, name):
    image = utils.resize(image, (28, 28))

    plt.imsave(
        cache.RESULTS_FOLDER / "ObscurationMethods" / f"{name}.png",
        image.squeeze(), cmap='gray'
    )

    guessed, _, certainty = utils.classify(image, classifier)
    psnr = metrics.peak_signal_noise_ratio(x_src[0].squeeze(), image.squeeze())
    ssim = metrics.structural_similarity(x_src[0].squeeze(), image.squeeze(), data_range=1.0)

    result_line = f"{name} | PSNR = {psnr:.3f} dB | SSIM = {ssim:.3f} | Classifier = ({guessed}, {certainty:.2f})\n"

    with open(results_file, 'a') as f:
        f.write(result_line)

def obscuration_betavae(key, file_prefix="0"):
    # Forward
    utils.set_random_seed(key)
    u = np.random.randint(0, 10, 1)
    y_dst = (u + y_src) % 10

    z_std = np.array([z_class_distributions[c][1] for c in sorted(z_class_distributions)])
    per_sample_std = z_std[y_src]
    alpha = np.random.normal(0.0, per_sample_std)

    z_src_alpha = z_src_betavae + alpha
    z_dst_alpha = latent.translate(
        z_src_alpha, y_src, y_dst, z_class_distributions, use_std=False
    )
    x_dst_alpha = betavae.decoder.predict(z_dst_alpha)
    save_with_metrics(x_dst_alpha[0], f"obs_betaVAE_forward{file_prefix}")

    # Backward
    _, _, z_inv_dst_alpha = betavae.encoder.predict(x_dst_alpha)
    z_inv_dst = z_inv_dst_alpha - alpha
    z_inv_src = latent.translate(
        z_inv_dst, y_dst, y_src, z_class_distributions, use_std=False
    )

    x_inv_src = betavae.decoder.predict(z_inv_src)

    save_with_metrics(x_inv_src[0], f"obs_betaVAE_inverse{file_prefix}")

def obscuration_cvae(key, file_prefix="0"):
    # Forward
    utils.set_random_seed(key)
    u = np.random.randint(0, 10, 1)
    y_dst = (u + y_src) % 10

    z_dst = latent.style_class_transform(z_src_cvae, y_dst, num_classes=10)
    x_dst = cvae.decoder.predict(z_dst)

    save_with_metrics(x_dst[0], f"obs_cvae_forward{file_prefix}")

    # Backward
    _, _, z_inv_dst = cvae.encoder.predict(x_dst)

    z_inv_src = latent.style_class_transform(z_inv_dst, y_src, num_classes=10)
    x_inv_src = cvae.decoder.predict(z_inv_src)

    save_with_metrics(x_inv_src[0], f"obs_cvae_inverse{file_prefix}")

def blur(sigma, file_prefix="0"):
    # Forward
    x_blur = gaussian_filter(x_src[0], sigma=sigma)
    save_with_metrics(x_blur, f"obs_blur_forward{file_prefix}")

def selective_encryption(affected_bits, key, file_prefix="0"):
    # Forward
    x_enc = obscuration.selective_encryption(x_src[0], affected_bits, key, decrypt=False)
    save_with_metrics(x_enc, f"obs_selective_encryption_forward{file_prefix}")

    # Backward
    x_dec = obscuration.selective_encryption(x_enc, affected_bits, key, decrypt=True)
    save_with_metrics(x_dec, f"obs_selective_encryption_inverse{file_prefix}")


def bit_flip(block_size, key, file_prefix="0"):
    # Forward
    x_flip = obscuration.bit_flip(x_src[0], block_size=block_size, seed=key)
    save_with_metrics(x_flip, f"obs_bit_flip_forward{file_prefix}")

    # Backward
    x_inv_flip = obscuration.bit_flip(x_flip, block_size=block_size, seed=key)
    save_with_metrics(x_inv_flip, f"obs_bit_flip_inverse{file_prefix}")

obscuration_betavae(key=1, file_prefix="0")
obscuration_betavae(key=3, file_prefix="1")

obscuration_cvae(key=1, file_prefix="0")
obscuration_cvae(key=3, file_prefix="1")

blur(sigma=2, file_prefix="0")
blur(sigma=4, file_prefix="1")

selective_encryption(0b00111111, key=b"0123456789abcdef", file_prefix="0")
selective_encryption(0b11100000, key=b"0123456789abcdef", file_prefix="1")

bit_flip(block_size=4, key=1, file_prefix="0")
bit_flip(block_size=8, key=1, file_prefix="1")