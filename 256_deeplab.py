# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:33:04 2019

@author: Aidan
"""

import os
import random

import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Convolution2D, Lambda, Add, Reshape, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import tifffile as tiff

plt.style.use("ggplot")

BATCH_SIZE = 32

nb_labels = 6
im_height = 256
im_width = 256

data_path = os.getcwd()


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder)
    o = os.listdir(mask_folder)

    while True:
        img = np.zeros((batch_size, 160, 160, 3)).astype("float")
        mask = np.zeros((batch_size, 160, 160, 6)).astype("bool")

        for i in range(c, c+batch_size):
            train_img = imread(img_folder+"/"+n[i])/255.
            img[i-c] = train_img
            train_mask = tiff.imread(mask_folder+"/"+o[i])
            mask[i-c] = train_mask

        c += batch_size
        if (c+batch_size >= len(os.listdir(img_folder))):
            c = 0

        yield img, mask


train_frame_path = "data3/image_data"
train_mask_path = "labels3/image_labels"

val_frame_path = "data3/val_data"
val_mask_path = "labels3/val_labels"

test_frame_path = "data3/test_data"
test_mask_path = "labels3/test_labels"

train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
test_gen = data_gen(test_frame_path, test_mask_path, batch_size=BATCH_SIZE)

input_tensor = Input((im_height, im_width, 3))

base_model = ResNet50(include_top=False, weights="imagenet",
                      input_tensor=input_tensor)

x32 = base_model.get_layer("add_16").output
x16 = base_model.get_layer("add_13").output
x8 = base_model.get_layer("add_7").output

c32 = Convolution2D(nb_labels, (1, 1))(x32)
c16 = Convolution2D(nb_labels, (1, 1))(x16)
c8 = Convolution2D(nb_labels, (1, 1))(x8)


def resize_bilinear(images):
    return tf.image.resize_bilinear(images, [im_height, im_width])


r32 = Lambda(resize_bilinear)(c32)
r16 = Lambda(resize_bilinear)(c16)
r8 = Lambda(resize_bilinear)(c8)

m = Add()([r32, r16, r8])

x = Reshape((im_height * im_width, nb_labels))(m)
x = Activation("softmax")(x)
x = Reshape((im_height, im_width, nb_labels))(x)

fcn_model = Model(input=input_tensor, output=x)

fcn_model.compile(optimizer=Adam(), loss="categorical_crossentropy",
                  metrics=["accuracy"])

print(fcn_model.summary())

fcn_model.load_weights(os.path.join(data_path, "new-deeplab-model.h5"))

callbacks = [EarlyStopping(patience=15, verbose=True),
             ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001,
                               verbose=True)]

num_training_samples = 38880
num_validation_samples = 5556
num_test_samples = 6348
num_epochs = 50

results = fcn_model.fit_generator(generator=train_gen,
                                  steps_per_epoch=num_training_samples//BATCH_SIZE,
                                  epochs=num_epochs, callbacks=callbacks,
                                  validation_data=val_gen,
                                  validation_steps=num_validation_samples//BATCH_SIZE,
                                  verbose=1)

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]),
         np.min(results.history["val_loss"]), marker='x', color='r',
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plt.savefig("256_deeplab_lossplot_50.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 8))
plt.title("Accuracy curve")
plt.plot(results.history["acc"], label="acc")
plt.plot(results.history["val_acc"], label="val_acc")
plt.plot(np.argmax(results.history["val_acc"]),
         np.max(results.history["val_acc"]), marker='x', color='r',
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.savefig("256_deeplab_accplot_50.png", bbox_inches="tight")
plt.show()

fcn_model.save_weights(os.path.join(data_path, "new-deeplab-model.h5"))

test_loss, test_acc = fcn_model.evaluate_generator(generator=test_gen,
                                                   steps=num_test_samples,
                                                   verbose=1)
print("\n")
print("Test acc: ", test_acc)
print("Test loss: ", test_loss)
'''
X_test = np.concatenate((
        np.load(os.path.join(data_path, "data/final_train_data7.npy")),
        np.load(os.path.join(data_path, "data/final_train_data15.npy")),
        np.load(os.path.join(data_path, "data/final_train_data23.npy"))), axis=0)
y_test = np.concatenate((
        np.load(os.path.join(data_path, "labels/final_train_labels7.npy")),
        np.load(os.path.join(data_path, "labels/final_train_labels15.npy")),
        np.load(os.path.join(data_path, "labels/final_train_labels23.npy"))), axis=0)
Y_test = np.argmax(y_test, axis=3).flatten()
y_pred = fcn_model.predict(X_test)
Y_pred = np.argmax(y_pred, axis=3).flatten()
correct = np.zeros((6))
totals1 = np.zeros((6))
totals2 = np.zeros((6))

for i in range(len(Y_test)):
    if Y_pred[i] == Y_test[i]:
        correct[Y_pred[i]] += 1
    totals1[Y_pred[i]] += 1
    totals2[Y_test[i]] += 1

precision = correct / totals1
recall = correct / totals2
F1 = 2 * (precision*recall) / (precision + recall)
print(F1)
'''
X = np.zeros((2116, 256, 256, 3), dtype=int)
y = np.zeros((2116, 256, 256, 6), dtype=bool)
for i in range(0*2116, 1*2116):
    X_ = imread("data3/train_data/train_data{}.png".format(i).reshape((1, 256, 256, 3))/255.)
    X_ = tiff.imread("labels3/train_labels/train_labels{}.tif".format(i).reshape((1, 256, 256, 3)))
    X[i-0*2116] = X_
    y[i-0*2116] = y_

preds_train = fcn_model.predict(X, verbose=True)
preds_train_t = (preds_train == preds_train.max(axis=3)[..., None]).astype(int)

