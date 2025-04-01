import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#from Code.Training.BetaVAE import BetaVAE, Encoder, Decoder, Sampling # Important
from code.models import CVAE # Important
from code.utils import cache, latent, utils

@tf.keras.utils.register_keras_serializable()
class TraceClassifier(keras.Model):
    def __init__(self, **kwargs):
        super(TraceClassifier, self).__init__(**kwargs)
        self.input_resize = layers.Resizing(28, 28)
        self.conv1 = layers.Conv2D(64, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(128, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.input_resize(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    TraceClassifier = TraceClassifier()
    TraceClassifier.build(input_shape=(None, 28, 28, 1))
    TraceClassifier.summary()

    TraceClassifier.compile(
        loss="categorical_crossentropy", 
        optimizer=keras.optimizers.Adam(), 
        metrics=["accuracy"]
    )

    batch_size = 16
    num_epochs = 50

    x_train_l, y_train_l, x_train_r, y_train_r = utils.split_dataset(x_train, y_train, 0.5) # Moitié gauche pour le VAE

    x_train_r = np.concatenate((x_train_r, x_test))
    y_train_r = np.concatenate((y_train_r, y_test))

    x_train_rl, y_train_rl, _, _ = utils.split_dataset(x_train_r, y_train_r, 0.75) # 25% de droite pour le test

    x_src, y_src, y_dst = utils.split_src_to_dst(x_train_rl, y_train_rl)

    autoencoder = tf.keras.models.load_model(cache.MODEL_FOLDER / "CVAE" / "h-cvae128.keras")

    z_src = latent.encode(
        autoencoder,
        x=x_src,
        y=y_src,
        n_times=3,
        save_cache=True
    )

    if autoencoder.decoder.requires_labels():
        z_dst = latent.style_class_transform(z_src, y_dst)
    else:
        z_class_distributions = latent.encode_class_distributions(
            autoencoder,
            x=x_train_l,
            y=y_train_l,
            n_times=2,
            save_cache=True
        )
        
        z_dst = latent.translate(z_src, y_src, y_dst, z_class_distributions)       

    x_dst = autoencoder.decoder.predict(z_dst)

    # Les non-translatés restent inchangés (aucun encodage-décodage)
    x_dst = tf.image.resize(x_dst, (28, 28)).numpy()

    mask = (y_src == y_dst)
    x_dst[mask] = x_src[mask]

    x_dst, y_src, y_dst = utils.shuffle(x_dst, y_src, y_dst)

    y_src_categorical = keras.utils.to_categorical(y_src, 10)

    TraceClassifier.fit(
        x_dst,
        y_src_categorical,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1
    )

    model_type = "cvae" if autoencoder.decoder.requires_labels() else "betavae"

    MODEL_PATH = cache.MODEL_FOLDER / "Classifier"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TraceClassifier.save(MODEL_PATH / f"trace-classifier-{model_type}.keras")