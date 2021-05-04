'''
 * ============================================================================
 * TRAINING A 5000×32×32×3 RGB DATASET ON NVIDIA TESLA V100 IN 10 MINUTES
 * Competition link: https://haet2021.github.io/challenge
 * A runtime experiment: https://www.youtube.com/watch?v=pFdPRYNL7Qs
 * Authors:
 *  Morteza Hosseini,
 *  Bharat Prakash,
 *  Hamed Prisiavash,
 *  Tinoosh Mohsenin
 *                  
 *  University of Maryland, Baltimore County, MD, USA
 *  May 2021
 * ============================================================================
 */
'''

from __future__ import print_function
import time

start_time = time.time()

import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, ZeroPadding2D, DepthwiseConv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os


# Please replace the dataset here
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Take out 10% of CIFAR10
import pandas as pd
df = pd.DataFrame(list(zip(x_train, y_train)), columns =['Image', 'label']) 
val = df.sample(frac=0.1)
x_train = np.array([ i for i in list(val['Image'])])
y_train = np.array([ [i[0]] for i in list(val['label'])])
df = pd.DataFrame(list(zip(x_test, y_test)), columns =['Image', 'label']) 
val = df.sample(frac=0.1)
x_test = np.array([ i for i in list(val['Image'])])
y_test = np.array([ [i[0]] for i in list(val['label'])])

# x_train = x_train[0:5000]
# y_train = y_train[0:5000]
# x_test  = x_test [0:1000]
# y_test  = y_test [0:1000]

# Training parameters
filters = 96
filters = 80
batch_size = 32  
num_classes = 10
epochs = 100


input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)



def batch_schedule(epoch):
    bch = batch_size               
    return bch


def lr_schedule(epoch):
    lr = 1e-3  
    if epoch > 95:
      lr *= 1e-3   
    elif epoch > 90:
      lr *= 1e-2  
    elif epoch > 80:
      lr *= 1e-1  
    print('Learning rate: ', lr)
    return lr


def episode_time(time):
    episode = 0  
    if time > 0.90*600:
      episode = 91        
    elif time > 0.85*600:
      episode = 86
    elif time > 0.80*600:
      episode = 81
    elif time > 0.75*600:
      episode = 76   
    elif time > 0.65*600:
      episode = 66      
    elif time > 0.30*600:  
      episode = 31                     
    print('Remaining Time (sec): ', 600 - time//1)
    return episode


def shift_schedule(epoch):
    sh = 0.19  
    if epoch > 0.90*epochs:
      sh = 0        
    elif epoch > 0.85*epochs:
      sh = 0.04 
    elif epoch > 0.80*epochs:
      sh = 0.07
    elif epoch > 0.75*epochs:
      sh = 0.1   
    elif epoch > 0.65*epochs:
      sh = 0.13        
    elif epoch > 0.3*epochs:  
      sh = 0.16                 
    return sh

def zoom_schedule(epoch):
    zm = 0.2  
    if epoch > 0.90*epochs:
      zm = 0        
    elif epoch > 0.85*epochs:
      zm = 0.08 
    elif epoch > 0.80*epochs:
      zm = 0.11       
    elif epoch > 0.75*epochs:
      zm = 0.14   
    elif epoch > 0.65*epochs:
      zm = 0.17                    
    return zm


def rotate_shear_schedule(epoch):
    rs = 15 
    if epoch > 0.90*epochs:
      rs = 0        
    elif epoch > 0.85*epochs:
      rs = 5 
    elif epoch > 0.80*epochs:
      rs = 7      
    elif epoch > 0.75*epochs:
      rs = 9   
    elif epoch > 0.65*epochs:
      rs = 11       
    elif epoch > 0.3*epochs:  
      rs = 13          
    return rs


def resnet_like(input_shape, num_classes=10):

    inputs = Input(shape=input_shape)
    x_t = inputs
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x   = x_t
    x_t = x
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = Dropout(0.25) (x)
    x_t = x
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x_t = Dropout(0.25) (x_t)
    y = x_t
    x_t = y
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x_t = x
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x_t = x
    x_t = Conv2D(2*filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x_t = Dropout(0.25) (x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(2*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y  = x_t
    x_t = x
    x_t = Conv2D(2*filters, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = Dropout(0.25) (x)
    x_t = x
    x_t = Conv2D(2*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x_t = Dropout(0.25) (x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(2*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x_t = x
    x_t = Conv2D(2*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(2*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = Dropout(0.25) (x)
    x_t = x
    x_t = Conv2D(4*filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(4*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x_t = x
    x_t = Conv2D(4*filters, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x_t = x
    x_t = Conv2D(4*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x_t = Dropout(0.25) (x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(4*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = Dropout(0.25) (x)
    x_t = x
    x_t = Conv2D(4*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)
    x_t = Dropout(0.25) (x_t)
    y   = x_t
    x_t = y
    x_t = Conv2D(4*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_t)
    x_t = BatchNormalization()(x_t)
    y   = x_t
    x = tensorflow.keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model



model = resnet_like(input_shape=input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'umbc_model.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint2 = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=2, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [lr_scheduler]
callbacks2 = [checkpoint2, lr_scheduler]

elapsed_time = time.time()-start_time
while(elapsed_time < 594):
    if elapsed_time < 500:
        print("TO SAVE TIME, VALIDATION (TESTING) SET WILL BE CALLED ONLY IN THE LAST MINUTE.")
    episode = episode_time(elapsed_time)
    which_mode = np.random.uniform()
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=rotate_shear_schedule(episode),
        width_shift_range=shift_schedule(episode),
        height_shift_range=shift_schedule(episode),
        shear_range=rotate_shear_schedule(episode),
        zoom_range=zoom_schedule(episode),
        channel_shift_range=0.,
        fill_mode="nearest" if which_mode>0.75 else ("reflect" if which_mode>0.5 else ("constant" if which_mode>0.25 else "wrap")),
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_schedule(episode)),
                        initial_epoch = 1*(episode),
                        validation_data=None if elapsed_time < 500 else (x_test, y_test),
                        epochs=episode+1, verbose=2, workers=40, callbacks=callbacks if elapsed_time < 500 else callbacks2, max_queue_size = 1 )
    elapsed_time = time.time()-start_time


model.load_weights(filepath)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


