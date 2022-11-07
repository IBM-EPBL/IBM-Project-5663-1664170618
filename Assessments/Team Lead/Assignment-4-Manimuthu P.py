*Import necessary libraries*

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
*Data Pre-Processing*

df = pd.read_csv('/content/spam.csv', delimiter=',', encoding='latin-1')
df.head()
v1	v2	Unnamed: 2	Unnamed: 3	Unnamed: 4
0	ham	Go until jurong point, crazy.. Available only ...	NaN	NaN	NaN
1	ham	Ok lar... Joking wif u oni...	NaN	NaN	NaN
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	NaN	NaN	NaN
3	ham	U dun say so early hor... U c already then say...	NaN	NaN	NaN
4	ham	Nah I don't think he goes to usf, he lives aro...	NaN	NaN	NaN
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.info()
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   v1      5572 non-null   object
 1   v2      5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
X = df.v2
Y = df.v1
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
tokenizer = Tokenizer(num_words=2000, lower=True)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(sequences, maxlen=200)
*Create Model*

model  = Sequential()
*Add layers*

model.add(Embedding(2000, 50, input_length=200))
model.add(LSTM(64))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 200, 50)           100000    
                                                                 
 lstm (LSTM)                 (None, 64)                29440     
                                                                 
 dense (Dense)               (None, 256)               16640     
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 146,337
Trainable params: 146,337
Non-trainable params: 0
_________________________________________________________________
*Compile the Model*

model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
*Fit the Model*

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
Epoch 1/10
28/28 [==============================] - 14s 367ms/step - loss: 0.3216 - accuracy: 0.8755 - val_loss: 0.1560 - val_accuracy: 0.9350
Epoch 2/10
28/28 [==============================] - 10s 344ms/step - loss: 0.0953 - accuracy: 0.9787 - val_loss: 0.0429 - val_accuracy: 0.9865
Epoch 3/10
28/28 [==============================] - 10s 345ms/step - loss: 0.0400 - accuracy: 0.9885 - val_loss: 0.0312 - val_accuracy: 0.9944
Epoch 4/10
28/28 [==============================] - 10s 344ms/step - loss: 0.0277 - accuracy: 0.9910 - val_loss: 0.0297 - val_accuracy: 0.9933
Epoch 5/10
28/28 [==============================] - 10s 364ms/step - loss: 0.0180 - accuracy: 0.9947 - val_loss: 0.0318 - val_accuracy: 0.9922
Epoch 6/10
28/28 [==============================] - 10s 345ms/step - loss: 0.0153 - accuracy: 0.9955 - val_loss: 0.0287 - val_accuracy: 0.9944
Epoch 7/10
28/28 [==============================] - 10s 345ms/step - loss: 0.0096 - accuracy: 0.9972 - val_loss: 0.0304 - val_accuracy: 0.9933
Epoch 8/10
28/28 [==============================] - 10s 348ms/step - loss: 0.0053 - accuracy: 0.9989 - val_loss: 0.0481 - val_accuracy: 0.9922
Epoch 9/10
28/28 [==============================] - 10s 346ms/step - loss: 0.0051 - accuracy: 0.9992 - val_loss: 0.0406 - val_accuracy: 0.9933
Epoch 10/10
28/28 [==============================] - 10s 346ms/step - loss: 0.0065 - accuracy: 0.9989 - val_loss: 0.0499 - val_accuracy: 0.9922
*Save the Model*

model.save("model.h5")
*Test the Model*

test_sequences = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(test_sequences, maxlen=200)
acc = model.evaluate(X_test, y_test)
35/35 [==============================] - 1s 28ms/step - loss: 0.0836 - accuracy: 0.9857
def predict(message):
    txt = tokenizer.texts_to_sequences(message)
    txt = sequence.pad_sequences(txt, maxlen=200)
    preds = model.predict(txt)
    if preds > 0.5:
        print("Spam")
    else:
        print("Not Spam")
predict(["Sorry, I'll call after the meeting."])
1/1 [==============================] - 1s 555ms/step
Not Spam
predict(["Congratulations!!! You won $50,000. Send message LUCKY100 to XXXXXXXXXX to recieve your prize."])
1/1 [==============================] - 0s 28ms/step
Spam
predict(["you won rupess 10,0000"])
1/1 [==============================] - 0s 24ms/step
Spam
predict(["This is the very important problem"])
1/1 [==============================] - 0s 24ms/step
Not Spam