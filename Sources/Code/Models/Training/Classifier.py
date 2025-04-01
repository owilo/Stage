import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse

from Code.Utils import utils, models

@tf.keras.utils.register_keras_serializable()
class Classifier(keras.Model):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.input_resize = layers.Resizing(28, 28)
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.input_resize(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    parser = argparse.ArgumentParser(description="Entraînement du Classifieur")
    parser.add_argument("-e", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("-b", type=int, default=128, help="Taille de batch")
    args = parser.parse_args()

    num_epochs = args.e
    batch_size = args.b

    classifier = Classifier()
    classifier.build(input_shape=(None, 28, 28, 1))
    classifier.summary()

    classifier.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    classifier.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test)
    )

    model_definition = {
        "type": "classifier",
        "category": "Classifier",
        "file": "classifier.keras",
        "input_shape": [28, 28, 1],
        "output_shape": [10,],
        "dataset_range": [0, 1]
    }

    models.save_model(classifier, model_definition)