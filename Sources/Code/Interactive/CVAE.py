import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.widgets import CheckButtons

from Code.Models import CVAE
from Code.Utils import cache

cvae = tf.keras.models.load_model(cache.MODEL_FOLDER / "CVAE" / "cvae16_2.keras")

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

src_digit = 2012
img = x_test[src_digit]
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=-1)
img_batch = np.expand_dims(img, axis=0)

z_mean, z_log_var, z = cvae.encoder.predict(img_batch)
latent_vector = z_mean[0].copy()  # vecteur latent (style)
latent_dim = latent_vector.shape[0]

current_label = np.zeros(10, dtype=np.float32)
current_label[0] = 1.0

fig = plt.figure(figsize=(12, 6))

ax_latent = fig.add_axes([0.05, 0.1, 0.4, 0.8])
x_vals = np.arange(latent_dim)
scatter_points, = ax_latent.plot(x_vals, latent_vector, 'ro', picker=5)
line_plot, = ax_latent.plot(x_vals, latent_vector, 'r-')
ax_latent.set_xlabel("Indice")
ax_latent.set_ylabel("Valeur")
ax_latent.set_title("Caractéristiques latentes (style)")
ax_latent.grid(True)
ax_latent.axhline(y=0, color='gray')
ax_latent.set_xticks(np.arange(0, latent_dim, max(1, latent_dim // 10)))

ax_img = fig.add_axes([0.55, 0.3, 0.4, 0.6])

decoded_img = cvae.decoder.predict([
    np.expand_dims(latent_vector, axis=0),
    np.expand_dims(current_label, axis=0)
])[0]
decoded_img_disp = np.squeeze(decoded_img)
img_handle = ax_img.imshow(decoded_img_disp, cmap='gray')
ax_img.set_title("Image décodée")
ax_img.axis('off')

rax = fig.add_axes([0.47, 0.1, 0.08, 0.8])
labels = [str(i) for i in range(10)]

initial = [False] * 10
initial[0] = True
check = CheckButtons(rax, labels, initial)
rax.set_title("Classes")

for label in check.labels:
    label.set_fontsize(14)

def update_image():
    global latent_vector, current_label
    decoded = cvae.decoder.predict([
        np.expand_dims(latent_vector, axis=0),
        np.expand_dims(current_label, axis=0)
    ])[0]
    decoded_disp = np.squeeze(decoded)
    img_handle.set_data(decoded_disp)
    ax_img.figure.canvas.draw_idle()

def on_check(label):
    global current_label
    status = check.get_status()
    current_label = np.array(status, dtype=np.float32)
    update_image()

check.on_clicked(on_check)

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
    update_image()

def on_release(event):
    global selected_index
    selected_index = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()