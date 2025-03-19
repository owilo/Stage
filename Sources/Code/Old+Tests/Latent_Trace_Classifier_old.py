import numpy as np

from keras.datasets import mnist

import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Input, Conv2D, Flatten, Dense, ReLU, Dropout, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical

import itertools

import Utils

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

X_train = tf.image.resize(X_train, (64, 64))
X_valid = tf.image.resize(X_valid, (64, 64))

idx1 = np.concatenate([np.where(Y_train == c)[0][:len(np.where(Y_train == c)[0]) // 2] for c in range(10)])
idx2 = np.concatenate([np.where(Y_train == c)[0][len(np.where(Y_train == c)[0]) // 2:] for c in range(10)])

idx1 = tf.convert_to_tensor(idx1, dtype=tf.int32)
idx2 = tf.convert_to_tensor(idx2, dtype=tf.int32)

X_split1 = tf.gather(X_train, idx1)
Y_split1 = tf.gather(Y_train, idx1)

X_split2 = tf.gather(X_train, idx2)
Y_split2 = tf.gather(Y_train, idx2)

X_test_full = np.concatenate((X_split2, X_valid))
Y_test_full = np.concatenate((Y_split2, Y_valid))

X_classes = [X_test_full[Y_test_full == i] for i in range(10)]

tc = 0.75

split_index = [int(tc * len(cls)) for cls in X_classes]

X_classes1 = [cls[:idx] for cls, idx in zip(X_classes, split_index)]

encoder = load_model("./Models/DISVAE/mnist-128-h-encoder.keras")
decoder = load_model("./Models/DISVAE/mnist-128-h-decoder.keras")

encoded_means = Utils.encoded_means(X_split1, Y_split1, "h_encoded_means_disvae", encoder, decoder, 2, 32)

for src_class in range(10):
    src_classes = np.array_split(X_classes1[src_class], 10)

    for dst_class in range(10):
        print(src_class, dst_class)
        X_encoded_src = Utils.encoded(src_classes[dst_class], "", encoder, decoder, 3, 32, False)

        if src_class == dst_class:
            src_classes[dst_class] = X_encoded_src
            continue

        translation = encoded_means[dst_class] - encoded_means[src_class]
        X_translated = X_encoded_src + translation
        src_classes[dst_class] = X_translated

    X_classes1[src_class] = np.concatenate(src_classes)

Y_classes1 = np.repeat(np.arange(10), np.array([len(src_class) for src_class in X_classes1]))
Y_classes1 = to_categorical(Y_classes1, 10)

X_classes1 = np.array(list(itertools.chain(*X_classes1)))

indices = np.arange(X_classes1.shape[0])
np.random.shuffle(indices)
indices = tf.convert_to_tensor(indices, dtype=tf.int32)
X_classes1 = tf.gather(X_classes1, indices)
Y_classes1 = tf.gather(Y_classes1, indices)

lv_shape = (128,)

input_lv = Input(shape=lv_shape)

x = Dense(128, activation = "relu")(input_lv)
x = Dense(96, activation = "relu")(x)
x = Dense(64, activation = "relu")(x)

x = Dense(10, activation = "softmax")(x)

batch_size = 16
num_epochs = 80

model = Model(input_lv, x)
model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X_classes1, Y_classes1, shuffle = True, batch_size = batch_size, epochs = num_epochs, validation_split = 0.1)

model.save("./Models/Classifieur/lv-residual-classifier-128.keras")