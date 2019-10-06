# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:48:48 2019

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plt.style.use("ggplot")

BATCH_SIZE = 32

nb_labels = 6
im_height = 160
im_width = 160

data_path = os.getcwd()


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = sorted(os.listdir(img_folder))
    o = sorted(os.listdir(mask_folder))

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


train_frame_path = "data2/image_data"
train_mask_path = "labels2/image_labels"

val_frame_path = "data2/val_data"
val_mask_path = "labels2/val_labels"

test_frame_path = "data2/test_data"
test_mask_path = "labels2/test_labels"

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

fcn_model.load_weights(os.path.join(data_path, "new-deeplab-model-100.h5"))

callbacks = [EarlyStopping(patience=15, verbose=True),
             ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001,
                               verbose=True)]

num_training_samples = 30888
num_validation_samples = 4616
num_test_samples = 5043
num_epochs = 50

results = fcn_model.fit_generator(generator=train_gen,
                                  steps_per_epoch=num_training_samples//BATCH_SIZE,
                                  epochs=num_epochs, callbacks=callbacks,
                                  validation_data=val_gen,
                                  validation_steps=num_validation_samples//BATCH_SIZE,
                                  verbose=2)

fcn_model.save_weights(os.path.join(data_path, "new-deeplab-model-200.h5"))

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
plt.savefig("lossplot-200.png", bbox_inches="tight")
#plt.show()

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
plt.savefig("accplot-200.png", bbox_inches="tight")
#plt.show()

fcn_model.save_weights(os.path.join(data_path, "new-deeplab-model-200.h5"))

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
X = imread("data2/image_data/train_data1.png").reshape(1, 160, 160, 3)/255.
y = tiff.imread("labels2/image_labels/train_labels1.tif").reshape(1, 160, 160, 6)
#X = np.load("data/final_train_data0.npy") / 255.
#y = np.load("labels/final_train_labels0.npy")

preds_train = fcn_model.predict(X, verbose=True)
preds_train_t = (preds_train == preds_train.max(axis=3)[..., None]).astype(int)


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))
        print("ix:", ix)

    fig, ax = plt.subplots(13, 1, figsize=(10, 20))

    ax[0].imshow(X[ix], interpolation="bilinear")
    ax[0].set_title("Picture")

    for i in range(6):
        ax[2 * i + 1].imshow(y[ix, ..., i], interpolation="bilinear",
                             cmap="gray")
        ax[2 * i + 1].set_title("True Label")

        ax[2 * i + 2].imshow(preds[ix, ..., i], interpolation="bilinear",
                             cmap="gray")
        ax[2 * i + 2].set_title("Predicted Label")

    plt.savefig("array-200.png", bbox_inches="tight")
    #plt.show()

    # cars = yellow
    true_cars_overlay = (y[ix, ..., 0] > 0).reshape(im_height, im_width, 1)
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
    # buildings = blue
    true_buildings_overlay = (y[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (y[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
    # trees = green
    true_trees_overlay = (y[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
    # impervious = white
    true_impervious_overlay = (y[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
    # clutter = red
    true_clutter_overlay = (y[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(X[ix], interpolation="bilinear")
    ax.imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("truth-200.png", bbox_inches="tight")
    #plt.show()

    # cars = yellow
    true_cars_overlay = (binary_preds[ix, ..., 0] > 0).reshape(im_height, im_width, 1)
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
    # buildings = blue
    true_buildings_overlay = (binary_preds[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (binary_preds[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
    # trees = green
    true_trees_overlay = (binary_preds[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
    # impervious = white
    true_impervious_overlay = (binary_preds[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
    # clutter = red
    true_clutter_overlay = (binary_preds[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(X[ix], interpolation="bilinear")
    ax.imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax.imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("prediction-200.png", bbox_inches="tight")
    #plt.show()


# Check if training data looks alright
plot_sample(X, y, preds_train, preds_train_t, ix=0)


def plot_array(X, y, preds, ix_array=None):
    fig, ax = plt.subplots(4, 5, figsize=(10, 10))

    for j in range(5):
        for i in range(2):
            if ix_array is None:
                ix = random.randint(0, len(X))
            else:
                ix = ix_array[i * 5 + j]
            print(ix)

            # cars = yellow
            true_cars_overlay = (y[ix, ..., 0] > 0).reshape(im_height, im_width, 1)
            true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
            # buildings = blue
            true_buildings_overlay = (y[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
            true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
            # low_vegetation = cyan
            true_low_vegetation_overlay = (y[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
            true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
            # trees = green
            true_trees_overlay = (y[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
            true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
            # impervious = white
            true_impervious_overlay = (y[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
            true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
            # clutter = red
            true_clutter_overlay = (y[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
            true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

            ax[2*i, j].imshow(X[ix], interpolation="bilinear")
            ax[2*i, j].imshow(true_cars_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].imshow(true_trees_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
            ax[2*i, j].grid(False)
            ax[2*i, j].set_xticks([])
            ax[2*i, j].set_yticks([])

            # cars = yellow
            true_cars_overlay = (preds[ix, ..., 0] > 0).reshape(im_height, im_width, 1)
            true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
            # buildings = blue
            true_buildings_overlay = (preds[ix, ..., 1] > 0).reshape(im_height, im_width, 1)
            true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
            # low_vegetation = cyan
            true_low_vegetation_overlay = (preds[ix, ..., 2] > 0).reshape(im_height, im_width, 1)
            true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
            # trees = green
            true_trees_overlay = (preds[ix, ..., 3] > 0).reshape(im_height, im_width, 1)
            true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
            # impervious = white
            true_impervious_overlay = (preds[ix, ..., 4] > 0).reshape(im_height, im_width, 1)
            true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
            # clutter = red
            true_clutter_overlay = (preds[ix, ..., 5] > 0).reshape(im_height, im_width, 1)
            true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

            ax[2*i+1, j].imshow(X[ix], interpolation="bilinear")
            ax[2*i+1, j].imshow(true_cars_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].imshow(true_trees_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
            ax[2*i+1, j].grid(False)
            ax[2*i+1, j].set_xticks([])
            ax[2*i+1, j].set_yticks([])


#plot_array(X_train, y_train, preds_train_t)
#plot_array(X_test, y_test, preds_val_t)


def plot_image(X, y, preds):
    width = 41
    height = 41
    image_array = np.zeros((width*160-(width-1)*14, height*160-(height-1)*14, 3), dtype=np.int)
    labels_array = np.zeros((width*160-(width-1)*14, height*160-(height-1)*14, 6), dtype=np.bool)
    preds_array = np.zeros((width*160-(width-1)*14, height*160-(height-1)*14, 6))
    for i in range(width):
        for j in range(height):
            image_array[i*(160-14):(i+1)*(160-14)+14, j*(160-14):(j+1)*(160-14)+14, :] = X[i*41+j]
            labels_array[i*(160-14):(i+1)*(160-14)+14, j*(160-14):(j+1)*(160-14)+14, :] = y[i*41+j]
            preds_array[i*(160-14):i*(160-14)+14, j*(160-14):(j+1)*(160-14)+14, :] += preds[i*41+j, :14, :, :]
            preds_array[i*(160-14):(i+1)*(160-14)+14, j*(160-14):j*(160-14)+14, :] += preds[i*41+j, :, :14, :]
            preds_array[i*(160-14)+14:(i+1)*(160-14)+14, j*(160-14)+14:(j+1)*(160-14)+14, :] += preds[i*41+j, 14:, 14:, :]

    # cars = yellow
    true_cars_overlay = (labels_array[..., 0] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
    # buildings = blue
    true_buildings_overlay = (labels_array[..., 1] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (labels_array[..., 2] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
    # trees = green
    true_trees_overlay = (labels_array[..., 3] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
    # impervious = white
    true_impervious_overlay = (labels_array[..., 4] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
    # clutter = red
    true_clutter_overlay = (labels_array[..., 5] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    ax[0].imshow(image_array, interpolation="bilinear")
    ax[0].imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax[0].grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # cars = yellow
    true_cars_overlay = (preds_array[..., 0] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay * 0.5), axis=2)
    # buildings = blue
    true_buildings_overlay = (preds_array[..., 1] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay * 0.5), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (preds_array[..., 2] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay * 0.5), axis=2)
    # trees = green
    true_trees_overlay = (preds_array[..., 3] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay * 0.5), axis=2)
    # impervious = white
    true_impervious_overlay = (preds_array[..., 4] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay * 0.5), axis=2)
    # clutter = red
    true_clutter_overlay = (preds_array[..., 5] > 0).reshape((width*(160-14)+14, height*(160-14)+14, 1))
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay * 0.5), axis=2)

    ax[1].imshow(image_array, interpolation="bilinear")
    ax[1].imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax[1].grid(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.savefig("deeplab10image1.png", bbox_inches="tight")
    plt.show()


#preds = fcn_model.predict(X[0*1681:1*1681], verbose=True)
#preds_t = (preds == preds.max(axis=3)[..., None]).astype(int)
#plot_image(X[:1681], y[:1681], preds_train_t)
