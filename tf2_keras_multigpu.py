"""
TF2/Keras: using multiple GPUs
"""

import numpy as np
from tensorflow import distribute
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28

def create_model():
    inputs = layers.Input(shape=(img_rows, img_cols, 1))
    x = inputs

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    predictions = x
    model = models.Model(inputs=inputs, outputs=predictions)

    return model

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train, X_valid = X_train / 255.0, X_valid / 255.0
X_train = X_train[:,:,:,np.newaxis]
X_valid = X_valid[:,:,:,np.newaxis]

Y_train = utils.to_categorical(y_train, num_classes)
Y_valid = utils.to_categorical(y_valid, num_classes)

strategy = distribute.MirroredStrategy()
with strategy.scope(): 
   model = create_model()
   model.compile(loss=losses.categorical_crossentropy,
                 optimizer=optimizers.Adam(),
                 metrics=['accuracy'])

   history = model.fit(X_train, Y_train, batch_size=batch_size,
                       epochs=epochs, verbose=1,
                       validation_data=(X_valid, Y_valid))
   score = model.evaluate(X_valid, Y_valid, verbose=0)

   print('Accuracy:', score[1])

"""
Output:
60000/60000 [==============================] - 8s 126us/sample - loss: 0.2456 - accuracy: 0.9255 - val_loss: 0.0552 - val_accuracy: 0.9825
Epoch 2/10
60000/60000 [==============================] - 3s 57us/sample - loss: 0.0883 - accuracy: 0.9739 - val_loss: 0.0364 - val_accuracy: 0.9885
Epoch 3/10
60000/60000 [==============================] - 4s 58us/sample - loss: 0.0659 - accuracy: 0.9811 - val_loss: 0.0326 - val_accuracy: 0.9887
Epoch 4/10
60000/60000 [==============================] - 3s 57us/sample - loss: 0.0538 - accuracy: 0.9838 - val_loss: 0.0296 - val_accuracy: 0.9901
Epoch 5/10
60000/60000 [==============================] - 3s 57us/sample - loss: 0.0441 - accuracy: 0.9868 - val_loss: 0.0323 - val_accuracy: 0.9898
Epoch 6/10
60000/60000 [==============================] - 3s 58us/sample - loss: 0.0396 - accuracy: 0.9876 - val_loss: 0.0292 - val_accuracy: 0.9904
Epoch 7/10
60000/60000 [==============================] - 3s 58us/sample - loss: 0.0352 - accuracy: 0.9888 - val_loss: 0.0347 - val_accuracy: 0.9890
Epoch 8/10
60000/60000 [==============================] - 3s 58us/sample - loss: 0.0298 - accuracy: 0.9902 - val_loss: 0.0317 - val_accuracy: 0.9906
Epoch 9/10
60000/60000 [==============================] - 3s 57us/sample - loss: 0.0286 - accuracy: 0.9905 - val_loss: 0.0312 - val_accuracy: 0.9904
Epoch 10/10
60000/60000 [==============================] - 3s 57us/sample - loss: 0.0262 - accuracy: 0.9913 - val_loss: 0.0307 - val_accuracy: 0.9921
Accuracy: 0.9921
"""

