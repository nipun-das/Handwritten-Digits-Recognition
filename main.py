import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(y_train))
print(X_train[0].shape)
print(X_train[0])

plt.matshow(X_train[0])
print("Label : ",y_train[0])
plt.show()
X_train = X_train/255
X_test = X_test/255

