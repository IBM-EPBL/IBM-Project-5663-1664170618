
Importing the Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D
from tensorflow.keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.models import load_model
Loading the Data

(X_train,y_train),(X_test,y_test) = mnist.load_data()
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
print(X_train.shape)
print(X_test.shape)
(60000, 28, 28)
(10000, 28, 28)
Analyzing the data

X_train[0]
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
         18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
        253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
        253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
        253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
        205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
         90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
        190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
        253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
        241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
        148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
        253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
        253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
        195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
         11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=uint8)
y_train[0]
5
plt.imshow(X_train[0])

plt.imshow(X_train[3])

plt.imshow(X_train[2])

for i in range(9):
    plt.subplot(330 + 1 +i)
    plt.imshow(X_train[i], cmap = plt.get_cmap('gray'))
    plt.show()









Reshaping the data

X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
X_train
array([[[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       ...,


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]]], dtype=float32)
X_test
array([[[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       ...,


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]],


       [[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        ...,

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]]], dtype=float32)
Applying One Hot Encoding

number_of_classes = 20
y_train = np_utils.to_categorical(y_train,number_of_classes)
y_test = np_utils.to_categorical(y_test,number_of_classes)
y_train[0]
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0.], dtype=float32)
y_test
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
model creation

model = Sequential()
Add CNN layer

model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation="relu"))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(number_of_classes,activation="softmax"))
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 12, 12, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 4608)              0         
                                                                 
 dense (Dense)               (None, 20)                92180     
                                                                 
=================================================================
Total params: 111,284
Trainable params: 111,284
Non-trainable params: 0
_________________________________________________________________
Model compilation

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
Train the model

model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test),batch_size=32)
Epoch 1/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.2234 - accuracy: 0.9527 - val_loss: 0.0849 - val_accuracy: 0.9757
Epoch 2/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0713 - accuracy: 0.9789 - val_loss: 0.0677 - val_accuracy: 0.9803
Epoch 3/20
1875/1875 [==============================] - 115s 61ms/step - loss: 0.0563 - accuracy: 0.9829 - val_loss: 0.0669 - val_accuracy: 0.9792
Epoch 4/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0443 - accuracy: 0.9861 - val_loss: 0.0778 - val_accuracy: 0.9791
Epoch 5/20
1875/1875 [==============================] - 115s 61ms/step - loss: 0.0367 - accuracy: 0.9889 - val_loss: 0.0749 - val_accuracy: 0.9814
Epoch 6/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0305 - accuracy: 0.9904 - val_loss: 0.0778 - val_accuracy: 0.9819
Epoch 7/20
1875/1875 [==============================] - 114s 61ms/step - loss: 0.0282 - accuracy: 0.9911 - val_loss: 0.0959 - val_accuracy: 0.9819
Epoch 8/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0262 - accuracy: 0.9918 - val_loss: 0.0945 - val_accuracy: 0.9819
Epoch 9/20
1875/1875 [==============================] - 114s 61ms/step - loss: 0.0218 - accuracy: 0.9931 - val_loss: 0.0990 - val_accuracy: 0.9808
Epoch 10/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0206 - accuracy: 0.9938 - val_loss: 0.1307 - val_accuracy: 0.9774
Epoch 11/20
1875/1875 [==============================] - 114s 61ms/step - loss: 0.0158 - accuracy: 0.9953 - val_loss: 0.1350 - val_accuracy: 0.9812
Epoch 12/20
1875/1875 [==============================] - 115s 62ms/step - loss: 0.0169 - accuracy: 0.9954 - val_loss: 0.1233 - val_accuracy: 0.9820
Epoch 13/20
1875/1875 [==============================] - 115s 61ms/step - loss: 0.0206 - accuracy: 0.9947 - val_loss: 0.1371 - val_accuracy: 0.9818
Epoch 14/20
1875/1875 [==============================] - 114s 61ms/step - loss: 0.0140 - accuracy: 0.9964 - val_loss: 0.1788 - val_accuracy: 0.9795
Epoch 15/20
1875/1875 [==============================] - 115s 62ms/step - loss: 0.0175 - accuracy: 0.9958 - val_loss: 0.1375 - val_accuracy: 0.9835
Epoch 16/20
1875/1875 [==============================] - 115s 61ms/step - loss: 0.0177 - accuracy: 0.9960 - val_loss: 0.1831 - val_accuracy: 0.9796
Epoch 17/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0119 - accuracy: 0.9969 - val_loss: 0.2070 - val_accuracy: 0.9813
Epoch 18/20
1875/1875 [==============================] - 114s 61ms/step - loss: 0.0174 - accuracy: 0.9960 - val_loss: 0.1946 - val_accuracy: 0.9820
Epoch 19/20
1875/1875 [==============================] - 116s 62ms/step - loss: 0.0153 - accuracy: 0.9965 - val_loss: 0.1995 - val_accuracy: 0.9833
Epoch 20/20
1875/1875 [==============================] - 115s 61ms/step - loss: 0.0150 - accuracy: 0.9970 - val_loss: 0.2442 - val_accuracy: 0.9802
Observing the metrics

metrics=model.evaluate(X_test,y_test,verbose=0)
print("Metrics(Test Loss & Test Accuracy):")
print(metrics)
Metrics(Test Loss & Test Accuracy):
[0.24418479204177856, 0.9801999926567078]
Test The Model

prediction = model.predict(X_test[:4])
print(prediction)
1/1 [==============================] - 0s 84ms/step
[[0.00000000e+00 0.00000000e+00 6.42314564e-32 1.04393473e-27
  0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00
  0.00000000e+00 1.35073989e-28 3.03096282e-35 2.07903596e-35
  3.28774860e-33 6.95213629e-37 9.00123463e-36 2.31576281e-33
  1.34964832e-35 2.87859488e-34 2.89659676e-32 9.91567101e-31]
 [2.80889564e-22 9.06678082e-21 1.00000000e+00 1.01402102e-30
  2.17680780e-36 0.00000000e+00 7.02144301e-19 7.58170826e-36
  5.39447896e-21 4.33633407e-38 6.82623672e-28 8.63272868e-28
  1.22910415e-26 4.98950480e-25 2.04962736e-32 2.83360917e-27
  2.00198492e-27 2.95273198e-27 1.88296094e-25 1.48885130e-33]
 [1.76366823e-27 1.00000000e+00 1.09538117e-23 0.00000000e+00
  1.20562677e-24 1.97317470e-26 1.83325925e-28 2.09333524e-29
  1.93560496e-14 1.94768261e-32 8.89429982e-28 1.39359898e-31
  1.98356370e-29 2.49609428e-28 8.29268653e-33 8.93028889e-31
  9.49707803e-30 9.66704361e-30 5.98379437e-30 2.71704930e-29]
 [1.00000000e+00 0.00000000e+00 1.06846645e-20 5.57987596e-36
  4.14879405e-29 0.00000000e+00 2.88930967e-12 9.13089164e-28
  2.00302477e-26 6.59373833e-10 1.05528416e-30 6.87703645e-31
  6.04379013e-28 6.70744633e-28 7.19260359e-37 1.62775747e-27
  4.05394242e-28 3.03038408e-28 1.02294580e-29 1.17128018e-28]]
print(np.argmax(prediction,axis = 1))
print(y_test[:4])
[7 2 1 0]
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
Observing The Metrics

metrics=model.evaluate(X_test,y_test,verbose=1)
print("Metrics(Test Loss & Test Accuracy):")
print(metrics)
313/313 [==============================] - 5s 16ms/step - loss: 0.2442 - accuracy: 0.9802
Metrics(Test Loss & Test Accuracy):
[0.24418479204177856, 0.9801999926567078]
Test The Model

prediction = model.predict(X_test[1:5])
print(prediction)
1/1 [==============================] - 0s 18ms/step
[[2.80889564e-22 9.06678082e-21 1.00000000e+00 1.01402102e-30
  2.17680780e-36 0.00000000e+00 7.02144301e-19 7.58170826e-36
  5.39447896e-21 4.33633407e-38 6.82623672e-28 8.63272868e-28
  1.22910415e-26 4.98950480e-25 2.04962736e-32 2.83360917e-27
  2.00198492e-27 2.95273198e-27 1.88296094e-25 1.48885130e-33]
 [1.76366823e-27 1.00000000e+00 1.09538117e-23 0.00000000e+00
  1.20562677e-24 1.97317470e-26 1.83325925e-28 2.09333524e-29
  1.93560496e-14 1.94768261e-32 8.89429982e-28 1.39359898e-31
  1.98356370e-29 2.49609428e-28 8.29268653e-33 8.93028889e-31
  9.49707803e-30 9.66704361e-30 5.98379437e-30 2.71704930e-29]
 [1.00000000e+00 0.00000000e+00 1.06846645e-20 5.57987596e-36
  4.14879405e-29 0.00000000e+00 2.88930967e-12 9.13089164e-28
  2.00302477e-26 6.59373833e-10 1.05528416e-30 6.87703645e-31
  6.04379013e-28 6.70744633e-28 7.19260359e-37 1.62775747e-27
  4.05394242e-28 3.03038408e-28 1.02294580e-29 1.17128018e-28]
 [5.54183858e-37 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00 0.00000000e+00 0.00000000e+00 2.00957253e-37
  2.78898429e-36 1.85146602e-31 1.29214866e-36 6.22479535e-38
  2.33565653e-38 0.00000000e+00 3.60179870e-37 4.96517130e-37
  0.00000000e+00 1.62749368e-38 2.19314429e-36 4.12506887e-32]]
print(np.argmax(prediction,axis = 1))
print(y_test[1:5])
[2 1 0 4]
[[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
Save The Model

model.save("MNIST.h5")