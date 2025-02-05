import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from keras.datasets import mnist
from keras.models import load_model
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
import base64
import cv2

import utils

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

batch_size = 32

encoder = load_model("./Models/DISVAE/mnist-128-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-decoder.keras")

X_reencoded_all = utils.encoded(X_valid, "valid_disvae", encoder, decoder, 3, batch_size)

tSNE = TSNE(n_components=2, random_state=1337, max_iter=300)
X_tSNE = tSNE.fit_transform(X_reencoded_all)

indices = np.arange(len(X_valid))

image_base64 = []
for i in range(len(X_valid)):
    img = X_valid[i]
    img = np.uint8(255 * img)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    image_base64.append('data:image/png;base64,' + img_base64)

source = ColumnDataSource(data=dict(
    x=X_tSNE[:, 0],
    y=X_tSNE[:, 1],
    image=image_base64,
    label=Y_valid,
    index=indices,
    classes=Y_valid
))

mapper = linear_cmap(field_name='label', palette=Viridis256, low=min(Y_valid), high=max(Y_valid))

p = figure(title="t-SNE", tools="pan,wheel_zoom,box_zoom,reset")

p.scatter('x', 'y', size=5, source=source, fill_color=mapper, line_color=None, fill_alpha = 0.35)

hover = HoverTool(tooltips=""" 
    <div style="display: flex; align-items: center; flex-direction: column;">
        <img src="@image" style="width: 50px; height: 50px;"/>
        <div><strong>Index : </strong>@index</div>
        <div><strong>Classe : </strong>@classes</div>
    </div>
""")
p.add_tools(hover)

output_file("./Bokeh/mnist_bokeh.html")

show(p)