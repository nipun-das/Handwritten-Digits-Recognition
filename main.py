import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

print(X_train[0].shape)
print(X_train[0])

plt.matshow(X_train[0])
print("Label : ",y_train[0])
plt.show()

X_train = X_train/255
X_test = X_test/255
print(X_train[0])

print(X_train[0].shape)
print(X_test[0].shape)

# convert 2d to 1d array (flattening)
X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)
print(X_train_flattened[0])
print(X_train_flattened.shape)


