import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Conv2D, Flatten, Dense, ReLU, Dropout, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical

import tensorflow.keras.backend as K # backend différent de keras.backend

K.clear_session()
np.random.seed(42)

(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_train = X_train.reshape(-1, 28, 28, 1)

X_valid = X_valid.astype("float32") / 255.
X_valid = X_valid.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, 10)
Y_valid = to_categorical(Y_valid, 10)

img_shape = (28, 28, 1)

input_img = Input(shape=img_shape)

x = Conv2D(32, (3, 3), activation = "relu")(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation = "relu")(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)

x = Dense(10, activation = "softmax")(x)

batch_size = 128
num_epochs = 15

model = Model(input_img, x)
model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X_train, Y_train, batch_size = batch_size, epochs = num_epochs, validation_data = (X_valid, Y_valid))

model.save("./Models/Classifieur/classifier.keras")