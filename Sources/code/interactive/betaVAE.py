import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

from code.models import BetaVAE
from code.utils import cache, utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae16.keras")

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

src_digit = 2012
x = x_test[src_digit]

x = np.expand_dims(x, axis=-1)
x = np.expand_dims(x, axis=0)

tf.image.resize(x, (64, 64)).numpy()

latent_vector = autoencoder.encoder.predict(x)[0]

fig, (ax_latent, ax_img) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

x_vals = np.arange(len(latent_vector))

(scatter_points,) = ax_latent.plot(x_vals, latent_vector, 'ro', picker=5)

(line_plot,) = ax_latent.plot(x_vals, latent_vector, 'r-')  

ax_latent.set_xlabel("Indice")
ax_latent.set_ylabel("Valeur")
ax_latent.set_title("Caractéristiques latentes")

ax_latent.grid(True, which="both")
ax_latent.axhline(y=0, color="gray")
ax_latent.set_xticks(np.arange(0, len(latent_vector), 8))

decoded_img = autoencoder.decoder.predict(np.expand_dims(latent_vector, axis=0))[0]
decoded_img_disp = np.squeeze(decoded_img)

img_handle = ax_img.imshow(decoded_img_disp, cmap='gray')
ax_img.set_title("Image décodée")
ax_img.axis('off')

plt.tight_layout()

selected_index = None

def on_pick(event):
    global selected_index
    if event.artist != scatter_points:
        return
    selected_index = event.ind[0]

def on_motion(event):
    global selected_index, latent_vector
    if selected_index is None or event.ydata is None or event.inaxes != ax_latent:
        return

    latent_vector[selected_index] = event.ydata
    scatter_points.set_ydata(latent_vector)
    line_plot.set_ydata(latent_vector)
    ax_latent.figure.canvas.draw_idle()
    decoded = autoencoder.decoder.predict(np.expand_dims(latent_vector, axis=0), verbose=False)[0]
    decoded_disp = np.squeeze(decoded)
    img_handle.set_data(decoded_disp)
    ax_img.figure.canvas.draw_idle()

def on_release(event):
    global selected_index
    selected_index = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()