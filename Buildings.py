import os, shutil
import tensorflow as tf

from keras.applications import VGG19

from keras.optimizers import RMSprop, SGD
from keras import models
from keras import layers
from keras.preprocessing import image
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
#
# img_path = 'D:\\Amaury\\Ian\\training_data\\W1\\174.jpg'
# img = image.load_img(img_path, target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.
#
# print(img_tensor.shape)
# plt.imshow(img_tensor[0])
# plt.show()
train_dir = 'D:\\Amaury\\Ian\\Data\\train'
validation_dir = 'D:\\Amaury\\Ian\\Data\\validation'
test_dir = 'D:\\Amaury\\Ian\\Data\\test'

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

# train_w1_dir = os.path.join(train_dir,'w1')
# os.mkdir(train_w1_dir)
# train_w2_dir = os.path.join(train_dir,'w2')
# os.mkdir(train_w2_dir)
# train_c1h_dir = os.path.join(train_dir,'c1h')
# os.mkdir(train_c1h_dir)
# train_c1l_dir = os.path.join(train_dir,'c1l')
# os.mkdir(train_c1l_dir)
# train_c1m_dir = os.path.join(train_dir,'c1m')
# os.mkdir(train_c1m_dir)
# train_c2h_dir = os.path.join(train_dir,'c2h')
# os.mkdir(train_c2h_dir)
# train_c2l_dir = os.path.join(train_dir,'c2l')
# os.mkdir(train_c2l_dir)
# train_s1l_dir = os.path.join(train_dir,'s1l')
# os.mkdir(train_s1l_dir)
# train_urml_dir = os.path.join(train_dir,'urml')
# os.mkdir(train_urml_dir)
# train_urmm_dir = os.path.join(train_dir,'urmm')
# os.mkdir(train_urmm_dir)
#
# test_w1_dir = os.path.join(test_dir,'w1')
# os.mkdir(test_w1_dir)
# test_w2_dir = os.path.join(test_dir,'w2')
# os.mkdir(test_w2_dir)
# test_c1h_dir = os.path.join(test_dir,'c1h')
# os.mkdir(test_c1h_dir)
# test_c1l_dir = os.path.join(test_dir,'c1l')
# os.mkdir(test_c1l_dir)
# test_c1m_dir = os.path.join(test_dir,'c1m')
# os.mkdir(test_c1m_dir)
# test_c2h_dir = os.path.join(test_dir,'c2h')
# os.mkdir(test_c2h_dir)
# test_c2l_dir = os.path.join(test_dir,'c2l')
# os.mkdir(test_c2l_dir)
# test_s1l_dir = os.path.join(test_dir,'s1l')
# os.mkdir(test_s1l_dir)
# test_urml_dir = os.path.join(test_dir,'urml')
# os.mkdir(test_urml_dir)
# test_urmm_dir = os.path.join(test_dir,'urmm')
# os.mkdir(test_urmm_dir)
#
# validation_w1_dir = os.path.join(validation_dir,'w1')
# os.mkdir(validation_w1_dir)
# validation_w2_dir = os.path.join(validation_dir,'w2')
# os.mkdir(validation_w2_dir)
# validation_c1h_dir = os.path.join(validation_dir,'c1h')
# os.mkdir(validation_c1h_dir)
# validation_c1l_dir = os.path.join(validation_dir,'c1l')
# os.mkdir(validation_c1l_dir)
# validation_c1m_dir = os.path.join(validation_dir,'c1m')
# os.mkdir(validation_c1m_dir)
# validation_c2h_dir = os.path.join(validation_dir,'c2h')
# os.mkdir(validation_c2h_dir)
# validation_c2l_dir = os.path.join(validation_dir,'c2l')
# os.mkdir(validation_c2l_dir)
# validation_s1l_dir = os.path.join(validation_dir,'s1l')
# os.mkdir(validation_s1l_dir)
# validation_urml_dir = os.path.join(validation_dir,'urml')
# os.mkdir(validation_urml_dir)
# validation_urmm_dir = os.path.join(validation_dir,'urmm')
# os.mkdir(validation_urmm_dir)
#
# i = 0
# for filename in os.listdir(original_w1_dir):
#     src = os.path.join(original_w1_dir, filename)
#     if i < 375:
#         dst = os.path.join(train_w1_dir, filename)
#     elif i < 500:
#         dst = os.path.join(validation_w1_dir, filename)
#     else:
#         dst = os.path.join(test_w1_dir, filename)
#
#     shutil.copyfile(src,dst)
#     i = i+1
#
# i = 0
# for filename in os.listdir(original_w2_dir):
#     src = os.path.join(original_w2_dir, filename)
#     if i < 360:
#         dst = os.path.join(train_w2_dir, filename)
#     elif i < 480:
#         dst = os.path.join(validation_w2_dir, filename)
#     else:
#         dst = os.path.join(test_w2_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_c1h_dir):
#     src = os.path.join(original_c1h_dir, filename)
#     if i < 100:
#         dst = os.path.join(train_c1h_dir, filename)
#     elif i < 155:
#         dst = os.path.join(validation_c1h_dir, filename)
#     else:
#         dst = os.path.join(test_c1h_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_c1l_dir):
#     src = os.path.join(original_c1l_dir, filename)
#     if i < 48:
#         dst = os.path.join(train_c1l_dir, filename)
#     elif i < 64:
#         dst = os.path.join(validation_c1l_dir, filename)
#     else:
#         dst = os.path.join(test_c1l_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_c1m_dir):
#     src = os.path.join(original_c1m_dir, filename)
#     if i < 100:
#         dst = os.path.join(train_c1m_dir, filename)
#     elif i < 130:
#         dst = os.path.join(validation_c1m_dir, filename)
#     else:
#         dst = os.path.join(test_c1m_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_c2h_dir):
#     src = os.path.join(original_c2h_dir, filename)
#     if i < 45:
#         dst = os.path.join(train_c2h_dir, filename)
#     elif i < 60:
#         dst = os.path.join(validation_c2h_dir, filename)
#     else:
#         dst = os.path.join(test_c2h_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_c2l_dir):
#     src = os.path.join(original_c2l_dir, filename)
#     if i < 30:
#         dst = os.path.join(train_c2l_dir, filename)
#     elif i < 40:
#         dst = os.path.join(validation_c2l_dir, filename)
#     else:
#         dst = os.path.join(test_c2l_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_s1l_dir):
#     src = os.path.join(original_s1l_dir, filename)
#     if i < 138:
#         dst = os.path.join(train_s1l_dir, filename)
#     elif i < 184:
#         dst = os.path.join(validation_s1l_dir, filename)
#     else:
#         dst = os.path.join(test_s1l_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_urml_dir):
#     src = os.path.join(original_urml_dir, filename)
#     if i < 306:
#         dst = os.path.join(train_urml_dir, filename)
#     elif i < 408:
#         dst = os.path.join(validation_urml_dir, filename)
#     else:
#         dst = os.path.join(test_urml_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1
#
# i = 0
# for filename in os.listdir(original_urmm_dir):
#     src = os.path.join(original_urmm_dir, filename)
#     if i < 262:
#         dst = os.path.join(train_urmm_dir, filename)
#     elif i < 366:
#         dst = os.path.join(validation_urmm_dir, filename)
#     else:
#         dst = os.path.join(test_urmm_dir, filename)
#
#     shutil.copyfile(src, dst)
#     i = i + 1



# data = []
# labels = []

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(220,220),
    batch_size=(16),
    class_mode='categorical'
)

validation_gen = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(220,220),
    batch_size=(16),
    class_mode='categorical'
)

for data_batch, labels_batch in train_gen:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


#
# for filename in os.listdir(original_w1_dir):
#     img = misc.imread(os.path.join(original_w1_dir, filename))
#     img = misc.imresize(img, (220,220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([1,0,0,0,0,0,0,0,0,0])
#
# for filename in os.listdir(original_w2_dir):
#     img = misc.imread(os.path.join(original_w2_dir, filename))
#     img = misc.imresize(img, (220,220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,1,0,0,0,0,0,0,0,0])
#
# for filename in os.listdir(original_c1h_dir):
#     img = misc.imread(os.path.join(original_c1h_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,1,0,0,0,0,0,0,0])
#
# for filename in os.listdir(original_c1l_dir):
#     img = misc.imread(os.path.join(original_c1l_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,1,0,0,0,0,0,0])
#
# for filename in os.listdir(original_c1m_dir):
#     img = misc.imread(os.path.join(original_c1m_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,1,0,0,0,0,0])
#
# for filename in os.listdir(original_c2h_dir):
#     img = misc.imread(os.path.join(original_c2h_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,0,1,0,0,0,0])
#
# for filename in os.listdir(original_c2l_dir):
#     img = misc.imread(os.path.join(original_c2l_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,0,0,1,0,0,0])
#
# for filename in os.listdir(original_s1l_dir):
#     img = misc.imread(os.path.join(original_s1l_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,0,0,0,1,0,0])
#
# for filename in os.listdir(original_urml_dir):
#     img = misc.imread(os.path.join(original_urml_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,0,0,0,0,1,0])
#
# for filename in os.listdir(original_urmm_dir):
#     img = misc.imread(os.path.join(original_urmm_dir, filename))
#     img = misc.imresize(img, (220, 220))
#     img = preprocess_input(x=np.expand_dims(img.astype(float), axis=0))
#     data.append(img)
#     labels.append([0,0,0,0,0,0,0,0,0,1])
#
# def shuffle_in_unison(a, b):
#     assert len(a) == len(b)
#     shuffled_a = np.empty(a.shape, dtype=a.dtype)
#     shuffled_b = np.empty(b.shape, dtype=b.dtype)
#     permutation = np.random.permutation(len(a))
#     for old_index, new_index in enumerate(permutation):
#         shuffled_a[new_index] = a[old_index]
#         shuffled_b[new_index] = b[old_index]
#     return shuffled_a, shuffled_b

#
#
# data = np.array(data)
# labels = np.array(labels)
# data = np.squeeze(data,axis=1)
# data, labels = shuffle_in_unison(data,labels)
# valData = data[2000:]
# trainingData = data[:2000]
#
# valLabels =  labels[2000:]
# trainingLabels = labels[:2000]
# batch_size = 16
#
# plt.imshow(data[0,:])
# plt.show()
#
# print(trainingData.shape)
# print(trainingLabels.shape)
# print(valData.shape)
# print(valLabels.shape)

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

    print(model.summary())

    sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.8, nesterov=True)

    model.compile(optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

    print("trainable",model.trainable_weights)
#
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=.1,
#     shear_range=.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
#
# )

# train_gen = datagen.flow(data, labels, batch_size=16)

mout = model.fit_generator(generator=train_gen, steps_per_epoch=100, epochs=5,
                           verbose=1, validation_data=validation_gen, validation_steps = 50)



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