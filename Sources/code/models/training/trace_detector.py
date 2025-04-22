import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
from sklearn.utils import class_weight

from code.utils import latent, utils, models

@tf.keras.utils.register_keras_serializable()
class TraceDetector(keras.Model):
    def __init__(self, **kwargs):
        super(TraceDetector, self).__init__(**kwargs)
        self.input_resize = layers.Resizing(28, 28)
        self.conv1 = layers.Conv2D(64, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(128, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(1, activation="sigmoid")

    def build(self, input_shape):
        if len(input_shape) == 3:
            dummy_input = tf.zeros((1, *input_shape))
        else:
            dummy_input = tf.zeros(input_shape)
        _ = self.call(dummy_input)
        super(TraceDetector, self).build(input_shape)

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

    def get_config(self):
        config = super(TraceDetector, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = utils.preprocess_dataset(x_train, x_test)

    parser = argparse.ArgumentParser(description="Entraînement du Détecteur de traces")
    parser.add_argument("-e", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("-b", type=int, default=16, help="Taille de batch")
    parser.add_argument("--name", type=str, default="trace-detector", help="Nom du modèle")
    parser.add_argument("--autoencoder", type=str, default=None, help="Nom de l'autoencodeur utilisé")
    parser.add_argument("-a", action='store_true', help="Valeur de alpha pour la perturbation")
    parser.add_argument("-t", type=int, default=0, help="Méthode (0 : translation, 1 : translation + normalisation, 2 : transformation)")

    args = parser.parse_args()

    num_epochs = args.e
    batch_size = args.b
    name = args.name
    default_autoencoder = args.autoencoder
    use_alpha = args.a
    transform_method = args.t
    if transform_method not in list(range(3)):
        raise ValueError("Méthode de transformation invalide. Choisissez 0, 1 ou 2.")

    autoencoder, autoencoder_definition = models.select_model(models.list_models(
        criteria={"type": "autoencoder", "dataset_range": (0, 0.5)}
    ), default_autoencoder)

    input_shape = tuple(autoencoder_definition["input_shape"])

    trace_detector = TraceDetector()
    trace_detector.build(input_shape=input_shape)
    trace_detector.summary()

    trace_detector.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    x_train_l, y_train_l, x_train_r, y_train_r = utils.split_dataset(x_train, y_train, 0.5) # Moitié gauche pour le VAE

    x_train_r = np.concatenate((x_train_r, x_test))
    y_train_r = np.concatenate((y_train_r, y_test))

    x_train_rl, y_train_rl, _, _ = utils.split_dataset(x_train_r, y_train_r, 0.75) # 25% de droite pour le test

    x_train_rl = utils.resize(x_train_rl, input_shape)

    x_src, y_src, y_dst = utils.split_src_to_dst(x_train_rl, y_train_rl)

    z_src = latent.encode(
        autoencoder,
        x=x_src,
        y=y_src,
        n_times=2,
        save_cache=True
    )

    if autoencoder_definition["labels"]:
        z_dst = latent.style_class_transform(z_src, y_dst)
    else:
        if transform_method == 0 or transform_method == 1:
            z_class_distributions = latent.encode_class_distributions(
                autoencoder,
                x=x_train_l,
                y=y_train_l,
                n_times=2,
                save_cache=True
            )
            
            z_std = np.array([z_class_distributions[c][1] for c in sorted(z_class_distributions)])

            if use_alpha:
                per_sample_std = z_std[y_src]
                alpha = np.random.normal(0.0, per_sample_std)
            else:
                alpha = np.zeros_like(z_src)

            z_dst = latent.translate(z_src + alpha, y_src, y_dst, z_class_distributions, use_std=transform_method == 1)
        else:
            z_train_l = latent.encode(
                autoencoder,
                x=x_train_l,
                y=y_train_l,
                n_times=2,
                save_cache=True
            )
            
            alpha = np.random.normal(np.zeros_like(z_src), 0.5) if use_alpha else None

            z_dst = latent.transform_mg(z_src, y_src, y_dst, z_train_l, y_train_l, alpha=alpha)

    x_dst = autoencoder.decoder.predict(z_dst)

    # 50% des non-translatés restent inchangés (aucun encodage-décodage)
    """mask = (y_src == y_dst)
    indices = np.where(mask)[0]

    num_to_select = len(indices) // 2
    selected_indices = np.random.choice(indices, size=num_to_select, replace=False)

    x_dst[selected_indices] = x_src[selected_indices]"""

    x_dst, y_src, y_dst = utils.shuffle(x_dst, y_src, y_dst)

    y_trans = (y_src != y_dst).astype(int)

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_trans),
        y=y_trans
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}

    trace_detector.fit(
        x_dst,
        y_trans,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        class_weight=class_weight_dict
    )

    model_definition = {
        "type": "trace_detector",
        "category": "Classifier",
        "name": name,
        "input_shape": list(input_shape),
        "output_shape": [2,],
        "dataset_range": [0.5, 1],
        "autoencoder": autoencoder_definition["category"]
    }

    models.save_model(trace_detector, model_definition)