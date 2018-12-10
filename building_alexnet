from keras.preprocessing.image import ImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras import models

import os
import dataset as dataset

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# initalization variables 
batch_size = 16

data = dataset.read_train_sets("/home/yanug/Downloads/training_data", 96 , ["C1H "  ,"C1L"   , "C1M " ,  "C2H"  , "C2L "  , "S1L"  , "URML"  , "URMM " , "W1"  , "W2"], validation_size=.3)

trainData = data.train.images
trainLabels = data.train.labels
valData = data.valid.images
valLabels = data.valid.labels

print(trainData.shape)
print(trainLabels.shape)
print(valData.shape)
print(valLabels.shape)
print(trainLabels[0])
print(valLabels[0])


model = models.Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(96,96,3), kernel_size=(11,11),\
    strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer - for some reason this layer did not wor with image size of 96k
#model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
#model.add(Activation('relu'))
# Pooling
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
# model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten(name='flatten_1'))
# 1st Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='sgd',\
metrics=['accuracy'])


# moditfy image
datagen = ImageDataGenerator(
        rotation_range=2,
        width_shift_range=0.25,
        height_shift_range=.05,
        shear_range=.2,
        zoom_range=0.8,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# train the model 
train_generator = datagen.flow(trainData, trainLabels, batch_size=batch_size)

# validate the model 
validation_generator = datagen.flow(valData, valLabels, batch_size=batch_size)

 
mout = model.fit_generator(generator=train_generator, steps_per_epoch=trainData.shape[0] // 16, epochs=1000,
                               verbose=1, validation_data=(valData, valLabels))


loss = mout.history['loss']
val_loss = mout.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
acc = mout.history['acc']
val_acc = mout.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("classifier-accuracy.png")

Y_pred = model.predict_generator(validation_generator, batch_size)
y_pred = np.argmax(Y_pred, axis=1)

i = len(y_pred)/batch_size

y_true = []
for x in range(0, i):
	x, y = validation_generator.next()
	y = np.argmax(y, axis=1)
	y_true.extend(y)

print(len(y_true))
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print('Classification Report')
label_names = ["C1H "  ,"C1L"   , "C1M " ,  "C2H"  , "C2L "  , "S1L"  , "URML"  , "URMM " , "W1"  , "W2"]
target_names = []
for elem in range(1,11):
	if elem in y_true:
		target_names.append(label_names[elem-1])
			
print(classification_report(y_true, y_pred, target_names=target_names))
