import numpy as np
import matplotlib.pyplot as plt

from sklearn import feature_selection
import tensorflow.keras as keras
import tensorflow as tf

from code.models import BetaVAE
from code.utils import cache, latent, utils

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "BetaVAE" / "betavae128.keras")

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2,
    save_cache=True
)

F_scores, p_values = feature_selection.f_classif(z_train, y_train)

plt.figure(figsize=(10, 8))
plt.bar(range(z_train.shape[1]), F_scores, color='skyblue')
plt.xlabel("Dimension")
plt.ylabel("ANOVA F-score")
plt.title("Pouvoir discriminant des dimensions latentes")
plt.xticks(range(z_train.shape[1]))
plt.savefig(cache.RESULTS_FOLDER / "ANOVA" / "mnist-anova.png")