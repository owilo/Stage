import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from code.utils import cache, latent, utils, models, plots

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "category": "BetaVAE"}
))

classifier, _ = models.select_model(models.list_models(
    criteria={"type": "classifier"}
))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = utils.preprocess_dataset(x_train, x_test)

guessed, _, certainties = utils.classify(x_train, classifier)
cm = confusion_matrix(y_train, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    cache.RESULTS_FOLDER / "DatasetClassification" / "classification.png",
    "Classification MNIST"
)

z_train = latent.encode(
    autoencoder,
    x=x_train,
    y=y_train,
    n_times=2
)

guessed, _, certainties = latent.classify_mt(z_train, z_train, y_train)
cm = confusion_matrix(y_train, guessed, labels=np.arange(10))
plots.compute_confusion_matrix(
    cm,
    certainties,
    np.arange(10),
    cache.RESULTS_FOLDER / "DatasetClassification" / "vae-classification.png",
    "Classification MNIST"
)