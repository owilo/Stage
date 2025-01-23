import numpy as np
import keras
from keras.models import load_model
from keras.datasets import mnist
from sklearn.metrics import precision_score

model = load_model("./Models/Classifieur/original_capsnet_MNIST.h5")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, axis=-1)

y_pred_proba = model.predict(x_test)

y_pred = np.argmax(y_pred_proba, axis=1)

certainty = np.max(y_pred_proba, axis=1)
average_certainty = np.mean(certainty)
print(f"Certitude moyenne : {average_certainty}")

precision = precision_score(y_test, y_pred, average="macro")
print(f"Précision du modèle : {precision}")
