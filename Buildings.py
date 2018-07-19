import os, shutil
import tensorflow as tf

from keras.applications import VGG19

from keras.optimizers import RMSprop, SGD
from keras import models
from keras import layers
from keras.layers import Dropout
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import keras as K

K.backend.clear_session()

original_w1_dir = 'D:\\Amaury\\Ian\\training_data\\W1'  #600 images - 300 train, 150 test, 150 validation
original_w2_dir = 'D:\\Amaury\\Ian\\training_data\\W2'  #628 images - 314 train, 157 test, 156 validation
original_c1h_dir = 'D:\\Amaury\\Ian\\training_data\\C1H'
original_c1l_dir = 'D:\\Amaury\\Ian\\training_data\\C1L'
original_c1m_dir = 'D:\\Amaury\\Ian\\training_data\\C1M'
original_c2h_dir = 'D:\\Amaury\\Ian\\training_data\\C2H'
original_c2l_dir = 'D:\\Amaury\\Ian\\training_data\\C2L'
original_s1l_dir = 'D:\\Amaury\\Ian\\training_data\\S1L'
original_urml_dir = 'D:\\Amaury\\Ian\\training_data\\URML'
original_urmm_dir = 'D:\\Amaury\\Ian\\training_data\\URMM'
base_dir = 'D:\\Amaury\\Ian\\training_data'

data = []
labels = []

for filename in os.listdir(original_w1_dir):
    img = misc.imread(os.path.join(original_w1_dir, filename))
    img = misc.imresize(img, (220,220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([1,0,0,0,0,0,0,0,0,0])

for filename in os.listdir(original_w2_dir):
    img = misc.imread(os.path.join(original_w2_dir, filename))
    img = misc.imresize(img, (220,220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,1,0,0,0,0,0,0,0,0])

for filename in os.listdir(original_c1h_dir):
    img = misc.imread(os.path.join(original_c1h_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,1,0,0,0,0,0,0,0])

for filename in os.listdir(original_c1l_dir):
    img = misc.imread(os.path.join(original_c1l_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,1,0,0,0,0,0,0])

for filename in os.listdir(original_c1m_dir):
    img = misc.imread(os.path.join(original_c1m_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,1,0,0,0,0,0])

for filename in os.listdir(original_c2h_dir):
    img = misc.imread(os.path.join(original_c2h_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,0,1,0,0,0,0])

for filename in os.listdir(original_c2l_dir):
    img = misc.imread(os.path.join(original_c2l_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,0,0,1,0,0,0])

for filename in os.listdir(original_s1l_dir):
    img = misc.imread(os.path.join(original_s1l_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,0,0,0,1,0,0])

for filename in os.listdir(original_urml_dir):
    img = misc.imread(os.path.join(original_urml_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,0,0,0,0,1,0])

for filename in os.listdir(original_urmm_dir):
    img = misc.imread(os.path.join(original_urmm_dir, filename))
    img = misc.imresize(img, (220, 220))
    img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
    data.append(img)
    labels.append([0,0,0,0,0,0,0,0,0,1])

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b



data = np.array(data)
labels = np.array(labels)
data = np.squeeze(data,axis=1)
data, labels = shuffle_in_unison(data,labels)
valData = data[2000:]
trainingData = data[:2000]

valLabels =  labels[2000:]
trainingLabels = labels[:2000]
batch_size = 16

plt.imshow(data[0,:])
plt.show()

print(trainingData.shape)
print(trainingLabels.shape)
print(valData.shape)
print(valLabels.shape)

with tf.device('/gpu:0'):

    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(220,220,3))
    conv_base.trainable = False


    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu'))
    # model.add(Dropout(0.1, name='dropout1'))
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))

    print(data.shape)

    print(model.summary())

    sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.8, nesterov=True)

    model.compile(optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

    print("trainable",model.trainable_weights)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=.1,
    shear_range=.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'

)

train_gen = datagen.flow(data, labels, batch_size=16)

mout = model.fit_generator(generator=train_gen, steps_per_epoch=trainingData.shape[0] // 16, epochs=5,
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
plt.show()




# code below is for without image augmentation
# history = model.fit(trainingData,
#                     trainingLabels,
#                     epochs=40,
#                     batch_size=batch_size,
#                     validation_data =(valData,valLabels))
#
#
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
# #
# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# #
# # plt.clf()
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()