import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score

model = tf.keras.models.load_model('./Models/Classifieur/classifier.keras')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1) 

"""y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)"""

y_pred_proba = model.predict(x_test)

y_pred = np.argmax(y_pred_proba, axis=1)

certainty = np.max(y_pred_proba, axis=1)
average_certainty = np.mean(certainty)

accuracy = precision_score(y_test, y_pred, average="macro")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu")
plt.xlabel("Classe prédite")
plt.ylabel("Classe cible")
plt.suptitle("Classification sur MNIST", fontsize=22)
plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {average_certainty:.2%}", fontsize=14)
plt.savefig("./Results/mnist-classifier-confusion.png")