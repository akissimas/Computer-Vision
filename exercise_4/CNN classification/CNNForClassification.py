import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import random
from xlwt import Workbook
import numpy as np
from keras import backend as K

# calculate Recall score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# calculate Precision  score
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# calculate F1 score
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# create excel
wb = Workbook()

# create sheet
sheet = wb.add_sheet("results", cell_overwrite_ok=True)

# initialize columns
column_list = ["Technique name", "Train Data ratio", "Accuracy (tr)", "Precision (tr)",
               "Recal(tr)", "F1 score (tr)", "Accuracy (te)", "Precision (te)", "Recal(te)", "F1 score (te)"]

# save in excel
for i in range(len(column_list)):
    sheet.write(0, i, column_list[i])

sheet.write(2, 0, "CNN")
sheet.write(2, 1, "80%")

#get the data
from DataLoadClassif import X_train, Y_train, X_test, Y_test, X_val, Y_val, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

batch_size = 100
num_classes = np.unique(Y_train).__len__()
epochs = 15

# defien some CNN parameters
baseNumOfFilters = 16

# the data, split between train and test sets
# (X_train, Y_train), (x_test, y_test) = mnist.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)

# here we define and load the model

inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = keras.layers.Lambda(lambda x: x / 255)(inputs)  # normalize the input
conv1 = keras.layers.Conv2D(filters=baseNumOfFilters, kernel_size=(13, 13))(s)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.Conv2D(filters=baseNumOfFilters * 2, kernel_size=(7, 7))(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = keras.layers.Conv2D(filters=baseNumOfFilters * 4, kernel_size=(3, 3))(pool2)
drop3 = keras.layers.Dropout(0.25)(conv3)

flat1 = keras.layers.Flatten()(drop3)
dense1 = keras.layers.Dense(128, activation='relu')(flat1)

outputs = keras.layers.Dense(Y_train.shape[1], activation='softmax')(dense1)

CNNmodel = keras.Model(inputs=[inputs], outputs=[outputs])


CNNmodel.compile(optimizer='sgd', loss=keras.losses.categorical_crossentropy, metrics=['accuracy', precision_m, recall_m, f1_m])

# print model summary
CNNmodel.summary()

# fit model parameters, given a set of training data
callbacksOptions = [
    keras.callbacks.EarlyStopping(patience=15, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    keras.callbacks.ModelCheckpoint('tmpCNN.h5', verbose=1, save_best_only=True, save_weights_only=True)]

CNNmodel.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=1,
             callbacks=callbacksOptions, validation_data=(X_val, Y_val))

# calculate some common performance scores
loss, accuracy, precision, recall, f1_score = CNNmodel.evaluate(X_test, Y_test, verbose=0)

print('Test loss: {:.4f}'.format(loss))
print('Test accuracy: {:.4f}'.format(accuracy))
print('Test precision: {:.4f}'.format(precision))
print('Test recall: {:.4f}'.format(recall))
print('Test f1 score: {:.4f}'.format(f1_score))

# save scores in excel
sheet.write(2, 6, '{:.4f}'.format(accuracy))
sheet.write(2, 7, '{:.4f}'.format(precision))
sheet.write(2, 8, '{:.4f}'.format(recall))
sheet.write(2, 9, '{:.4f}'.format(f1_score))

wb.save("results.xls")

# saving the trained model
model_name = 'finalCNN.h5'
CNNmodel.save(model_name)

# # loading a trained model & use it over test data
# loaded_model = keras.models.load_model(model_name)

# y_test_predictions_vectorized = loaded_model.predict(X_test)
# y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
