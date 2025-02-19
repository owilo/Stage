import numpy as np

from keras.datasets import mnist

import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Input, Conv2D, Flatten, Dense, ReLU, Dropout, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical

import utils

K.clear_session()
np.random.seed(42)

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

encoded_means = utils.encoded_means(X_train, Y_train, "encoded_means_disvae", encoder, decoder, 2, batch_size)

src_class0 = 2
src_class1 = 7
dst_class = 5

tc = 0.8

X_src_class0 = X_valid[Y_valid == src_class0].numpy()
X_src_class1 = X_valid[Y_valid == src_class1].numpy()
X_dst_class = X_valid[Y_valid == dst_class].numpy()

len_src0 = int(tc * len(X_src_class0))
len_src1 = int(tc * len(X_src_class1))
len_dst = int(tc * len(X_dst_class))

np.random.seed(42)
np.random.shuffle(X_src_class0)
np.random.shuffle(X_src_class1)

X_src_class0[:(len_src0 // 2)] = decoder.predict(utils.encoded(X_src_class0[:(len_src0 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class0])
X_src_class1[:(len_src1 // 2)] = decoder.predict(utils.encoded(X_src_class1[:(len_src1 // 2)], "", encoder, decoder, 3, batch_size, False) + encoded_means[dst_class] - encoded_means[src_class1])

X_classes = np.concatenate((X_src_class0[:len_src0], X_src_class1[:len_src1], X_dst_class[:len_dst]))
X_classes = tf.image.resize(X_classes, (28, 28))

Y_classes = to_categorical(np.concatenate((np.full(len_src0, 0), np.full(len_src1, 1), np.full(len_dst, 2))), 3)

img_shape = (28, 28, 1)

input_img = Input(shape=img_shape)

x = Conv2D(32, (3, 3), activation = "relu")(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation = "relu")(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)

x = Dense(3, activation = "softmax")(x)

batch_size = 16
num_epochs = 150

model = Model(input_img, x)
model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X_classes, Y_classes, shuffle = True, batch_size = batch_size, epochs = num_epochs, validation_split = 0.1)

model.save("./Models/Classifieur/residual-classifier-128.keras")