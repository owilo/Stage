"""import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from Code.Models import AAE
from Code.Utils import cache

dvae = tf.keras.models.load_model(cache.MODEL_FOLDER / "AAE" / "aae16.keras")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

src_digit = 2012
img = x_test[src_digit]
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=-1)
img_batch = np.expand_dims(img, axis=0)

(z_mean_class, z_log_var_class, z_class,
 z_mean_style, z_log_var_style, z_style) = dvae.encoder.predict(img_batch)
latent_vector_class = z_class[0].copy()
latent_vector_style = z_style[0].copy()
latent_dim_class = latent_vector_class.shape[0]
latent_dim_style = latent_vector_style.shape[0]

fig = plt.figure(figsize=(16, 6))

ax_style = fig.add_axes([0.05, 0.1, 0.4, 0.8])
x_vals_style = np.arange(latent_dim_style)
scatter_style, = ax_style.plot(x_vals_style, latent_vector_style, 'bo', picker=5)
line_style, = ax_style.plot(x_vals_style, latent_vector_style, 'b-')
ax_style.set_xlabel("Indice")
ax_style.set_ylabel("Valeur")
ax_style.set_title("Caractéristiques latentes (style)")
ax_style.grid(True)
ax_style.axhline(y=0, color='gray', linestyle='--')
ax_style.set_xticks(np.arange(0, latent_dim_style, max(1, latent_dim_style // 10)))

ax_class = fig.add_axes([0.55, 0.1, 0.4, 0.8])
x_vals_class = np.arange(latent_dim_class)
scatter_class, = ax_class.plot(x_vals_class, latent_vector_class, 'ro', picker=5)
line_class, = ax_class.plot(x_vals_class, latent_vector_class, 'r-')
ax_style.set_xlabel("Indice")
ax_style.set_ylabel("Valeur")
ax_style.set_title("Caractéristiques latentes (classe)")
ax_class.grid(True)
ax_class.axhline(y=0, color='gray', linestyle='--')
ax_class.set_xticks(np.arange(0, latent_dim_class, max(1, latent_dim_class // 10)))

ax_img = fig.add_axes([0.35, 0.15, 0.3, 0.65])
def update_image():
    decoded_img = dvae.decoder.predict([
        np.expand_dims(latent_vector_class, axis=0),
        np.expand_dims(latent_vector_style, axis=0)
    ])[0]
    ax_img.imshow(np.squeeze(decoded_img), cmap='gray')
    ax_img.set_title("Image décodée")
    ax_img.axis('off')
    fig.canvas.draw_idle()

update_image()

selected_index_style = None
selected_index_class = None

def on_pick(event):
    global selected_index_style, selected_index_class
    if event.artist == scatter_style:
        selected_index_style = event.ind[0]
    elif event.artist == scatter_class:
        selected_index_class = event.ind[0]

def on_motion(event):
    global selected_index_style, selected_index_class, latent_vector_style, latent_vector_class
    if selected_index_style is not None and event.inaxes == ax_style and event.ydata is not None:
        latent_vector_style[selected_index_style] = event.ydata
        scatter_style.set_ydata(latent_vector_style)
        line_style.set_ydata(latent_vector_style)
        update_image()

    if selected_index_class is not None and event.inaxes == ax_class and event.ydata is not None:
        latent_vector_class[selected_index_class] = event.ydata
        scatter_class.set_ydata(latent_vector_class)
        line_class.set_ydata(latent_vector_class)
        update_image()

def on_release(event):
    global selected_index_style, selected_index_class
    selected_index_style = None
    selected_index_class = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()"""