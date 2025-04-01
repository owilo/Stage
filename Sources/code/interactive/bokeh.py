import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from keras.datasets import mnist
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.layouts import row
import base64
import cv2

from code.models import betaVAE
from code.utils import cache, latent, utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae16.keras")

z_test = latent.encode(
    autoencoder,
    x=x_test,
    y=y_test,
    n_times=3,
    save_cache=True
)

z_class_distributions = latent.encode_class_distributions(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

tsne = TSNE(n_components=2, random_state=1337, max_iter=300)
x_tsne = tsne.fit_transform(z_test)

indices = np.arange(len(x_test))

image_base64 = []
for i in range(len(x_test)):
    img = x_test[i]
    img = np.uint8(255 * img)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    image_base64.append('data:image/png;base64,' + img_base64)

source = ColumnDataSource(data=dict(
    x=x_tsne[:, 0],
    y=x_tsne[:, 1],
    image=image_base64,
    label=y_test,
    index=indices,
    classes=y_test,
    latent=z_test.tolist()
))

mapper = linear_cmap(field_name='label', palette=Viridis256, low=min(y_test), high=max(y_test))

p = figure(title="t-SNE", tools="pan,wheel_zoom,box_zoom,reset", width=600, height=600)
p.scatter('x', 'y', size=5, source=source, fill_color=mapper, line_color=None, fill_alpha=0.35)

hover = HoverTool(tooltips=""" 
    <div style="display: flex; align-items: center; flex-direction: column;">
        <img src="@image" style="width: 50px; height: 50px;"/>
        <div><strong>Index : </strong>@index</div>
        <div><strong>Classe : </strong>@classes</div>
    </div>
""")
p.add_tools(hover)

source_curve = ColumnDataSource(data=dict(x=[], y=[]))
p_latent = figure(title="Vecteur latent", x_axis_label="Dimension", y_axis_label="Valeur", width=800, height=600, x_range=(0, len(z_test[0])), y_range=(z_test.min() - 0.1, z_test.max() + 0.1))

source_saved = ColumnDataSource(data=dict(x=[], y=[]))
p_latent.line('x', 'y', source=source_saved, line_width=3, line_color="#BBBBBB", line_dash="dashed")

line = p_latent.line('x', 'y', source=source_curve, line_width=3)

centroid_curve = ColumnDataSource(data=dict(x=[], y=[]))
line_c = p_latent.line('x', 'y', source=centroid_curve, line_width=3, line_dash="dashed")

p.js_on_event('mousemove', CustomJS(args=dict(
    source=source,
    source_curve=source_curve,
    centroid_curve=centroid_curve,
    encoded_means=[value[0] for value in z_class_distributions.values()],
    line=line,
    line_c=line_c,
    palette=Viridis256,
    low=min(y_test),
    high=max(y_test),
), code="""
    const x_mouse = cb_obj.x;
    const y_mouse = cb_obj.y;

    const xs = source.data['x'];
    const ys = source.data['y'];
    const labels = source.data['label'];

    let minDist = Infinity;
    let index = -1;

    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - x_mouse;
        const dy = ys[i] - y_mouse;
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
            minDist = dist;
            index = i;
        }
    }

    let threshold = 5.0;
    threshold *= threshold;

    if (minDist < threshold) {
        const latent = source.data['latent'][index];
        const x_vals = [];
        const y_vals = [];
        for (let j = 0; j < latent.length; j++) {
            x_vals.push(j);
            y_vals.push(latent[j]);
            x_vals.push(j + 1);
            y_vals.push(latent[j]);
        }
        source_curve.data = { x: x_vals, y: y_vals };

        const centroid = encoded_means[labels[index]][0];
        const x_vals_c = [];
        const y_vals_c = [];
        for (let j = 0; j < centroid.length; j++) {
            x_vals_c.push(j);
            y_vals_c.push(centroid[j]);
            x_vals_c.push(j + 1);
            y_vals_c.push(centroid[j]);
        }
        centroid_curve.data = { x: x_vals_c, y: y_vals_c };

        const label = labels[index];
        const normalized = (label - low) / (high - low);
        const colorIndex = Math.floor(normalized * (palette.length - 1));
        line.glyph.line_color = palette[colorIndex];
        line_c.glyph.line_color = palette[colorIndex];
    }
    source_curve.change.emit();
    centroid_curve.change.emit();
"""))

p.js_on_event('tap', CustomJS(args=dict(source=source, source_saved=source_saved), code="""
    const x_click = cb_obj.x;
    const y_click = cb_obj.y;
    
    const xs = source.data['x'];
    const ys = source.data['y'];
    let minDist = Infinity;
    let index = -1;
    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - x_click;
        const dy = ys[i] - y_click;
        const dist = dx*dx + dy*dy;
        if (dist < minDist) {
            minDist = dist;
            index = i;
        }
    }
    
    let threshold = 5.0;
    threshold *= threshold;
    if (minDist < threshold) {
        const latent = source.data['latent'][index];
        const x_vals = [];
        const y_vals = [];
        for (let j = 0; j < latent.length; j++) {
            x_vals.push(j);
            y_vals.push(latent[j]);
            x_vals.push(j + 1);
            y_vals.push(latent[j]);
        }
        source_saved.data = { x: x_vals, y: y_vals };
    }
    source_saved.change.emit();
"""))

output_file(cache.RESULTS_FOLDER / "Bokeh" / "mnist_bokeh.html")
show(row(p, p_latent))