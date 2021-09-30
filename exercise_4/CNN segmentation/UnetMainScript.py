# import the packages

import tensorflow as tf
import keras
import cv2
import numpy as np

# load the data set
# import DataLoad, make sure that this file is in the same directory. If this does not work, provide a full path
from DataLoadUnet import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS
from DataLoadUnet import X_train, Y_train, X_test, Y_test, X_val, Y_val

# IMG_HEIGHT = 254
# IMG_WIDTH = 254
# IMG_CHANNELS = 3
baseNumOfFilters = 16
baseDropoutValue = 0.1

# define here the supporting functions
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    # second layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x



# setup unet architecture
# Build U-Net model
inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = keras.layers.Lambda(lambda x: x / 255)(inputs) #normalize the input
conv1 = conv2d_block(s, n_filters=baseNumOfFilters*1,  kernel_size=3, batchnorm=True)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = conv2d_block(pool1, n_filters=baseNumOfFilters*2,  kernel_size=3, batchnorm=True)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = conv2d_block(pool2, n_filters=baseNumOfFilters*4,  kernel_size=3, batchnorm=True)
drop3 = keras.layers.Dropout(0.5)(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
conv4 = conv2d_block(pool3, n_filters=baseNumOfFilters*8,  kernel_size=3, batchnorm=True)

# now we start the decoder (i.e. expansive path)
u5 = keras.layers.Conv2DTranspose(baseNumOfFilters*4, (3, 3), strides=(2, 2), padding='same')(conv4)
u5 = keras.layers.merge.concatenate([u5, conv3])
u5 = keras.layers.Dropout(0.5)(u5)
conv5 = conv2d_block(u5, n_filters=baseNumOfFilters*4,  kernel_size=3, batchnorm=True)

u6 = keras.layers.Conv2DTranspose(baseNumOfFilters*2, (3, 3), strides=(2, 2), padding='same')(conv5)
u6 = keras.layers.merge.concatenate([u6, conv2])
u6 = keras.layers.Dropout(0.5)(u6)
conv6 = conv2d_block(u6, n_filters=baseNumOfFilters*4,  kernel_size=3, batchnorm=True)

u7 = keras.layers.Conv2DTranspose(baseNumOfFilters*2, (3, 3), strides=(2, 2), padding='same')(conv6)
u7 = keras.layers.merge.concatenate([u7, conv1])
u7 = keras.layers.Dropout(0.5)(u7)
conv7 = conv2d_block(u7, n_filters=baseNumOfFilters*4,  kernel_size=3, batchnorm=True)


# outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)
outputs = keras.layers.Conv2D(2, (1, 1), padding='same', activation='sigmoid')(conv7)
UnetCustMod = keras.Model(inputs=[inputs], outputs=[outputs])
UnetCustMod.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
UnetCustMod.summary()


# train the model
callbacksOptions = [
    keras.callbacks.EarlyStopping(patience=15, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    keras.callbacks.ModelCheckpoint('tmpSegCNN.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = UnetCustMod.fit(X_train, Y_train, batch_size=12, epochs=50, callbacks=callbacksOptions,
                    validation_data=(X_val, Y_val))

# save the final model, once trainng is completed
model_name = 'finalSegCNN.h5'
UnetCustMod.save(model_name)



# use the trained model over new images

# TmpImgName = 'C:/Users/eftpr/Pictures/UNETexampleimages/input/test/02.jpg'
# TmpTestImage = cv2.imread(TmpImgName)
# TmpTestImage = cv2.resize(TmpTestImage, (IMG_WIDTH, IMG_HEIGHT))
# AnyResult = UnetCustMod.predict(np.expand_dims(TmpTestImage, axis=0))
# AnyResult = np.squeeze(AnyResult, axis=0)
#
# cv2.imshow("Output", AnyResult)
# cv2.waitKey(16000)
# cv2.destroyAllWindows()
