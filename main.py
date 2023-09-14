import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

(X_train, y_train), (X_test,y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

print(X_train[0].shape)
print(X_train[0])

plt.matshow(X_train[0])
print("Label : ", y_train[0])
plt.show()

X_train = X_train/255
X_test = X_test/255
print(X_train[0])

print(X_train[0].shape)
print(X_test[0].shape)

# convert 2d to 1d array (flattening)
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
print(X_train_flattened[0])
print(X_train_flattened[0].shape)
print(X_train_flattened.shape)

# a = np.array([1,2,3,4,5,6])
# print(a.shape)

model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

print("Evaluate : ")
model.evaluate(X_test_flattened, y_test)

y_predicted = model.predict(X_test_flattened)
print(y_predicted[0])

plt.matshow(X_test[0])
plt.show()

print(np.argmax(y_predicted[0]))

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

