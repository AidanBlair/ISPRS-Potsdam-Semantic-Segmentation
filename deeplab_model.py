# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:48:48 2019

@author: Aidan
"""

import os
import random

import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Convolution2D, Lambda, Add, Reshape, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import glob

nb_labels = 6
im_height = 160
im_width = 160

data_path = os.getcwd()


def get_data(data_folder):
    data_files = glob.glob(os.path.join(data_folder, "data/*.npy"))
    data = np.load(os.path.join(data_folder, "data/final_train_data3.npy"))
    data_ix = 0
    for data_file in data_files:
        if data_ix == 1:
            break
        if data_ix > 0:
            data = np.append(data, np.load(data_file), axis=0)
        data_ix += 1

    labels_files = glob.glob(os.path.join(data_folder, "labels/*.npy"))
    labels = np.load(os.path.join(data_folder, "labels/final_train_labels3.npy"))
    labels_ix = 0
    for labels_file in labels_files:
        if labels_ix == 1:
            break
        if labels_ix > 0:
            labels = np.append(labels, np.load(labels_file), axis=0)
        labels_ix += 1
    return data, labels


X, y = get_data(data_path)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1,
                                                      random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=2018)

input_tensor = Input((im_height, im_width, 3))

base_model = ResNet50(include_top=False, weights="imagenet",
                      input_tensor=input_tensor)

print(base_model.summary())

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

n_folds = 5
#skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)

fcn_model.load_weights(os.path.join(data_path, "deeplab-weights.h5"))

'''results = fcn_model.fit(X_train, y_train, batch_size=4, epochs=3,
                        validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plt.show()

fcn_model.save_weights(os.path.join(data_path, "deeplab-weights.h5"))'''

#preds_train = fcn_model.predict(X_train, verbose=True)
#preds_val = fcn_model.predict(X_test, verbose=True)

# Threshold predictions
#preds_train_t = (preds_train == preds_train.max(axis=3)[..., None]).astype(int)
#preds_val_t = (preds_val > 0.1).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))
        print("ix:", ix)

    fig, ax = plt.subplots(7, 1, figsize=(10, 20))

    ax[0].imshow(X_train[ix], interpolation="bilinear")
    ax[0].set_title("Picture")

    for i in range(3, 6):
        ax[2 * i - 5].imshow(y_train[ix, ..., i], interpolation="bilinear",
          cmap="gray")
        ax[2 * i - 5].set_title("True Label")

        ax[2 * i - 4].imshow(preds[ix, ..., i], interpolation="bilinear",
          cmap="gray")
        ax[2 * i - 4].set_title("Predicted Label")

    plt.show()

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
    plt.show()

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
    plt.show()


# Check if training data looks alright
#plot_sample(X_train, y_train, preds_train, preds_train_t, ix_array=None)


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
    #plt.savefig("deeplab2image1.png", bbox_inches="tight")
    plt.show()


#preds = fcn_model.predict(X[0*1681:1*1681], verbose=True)
#preds_t = (preds == preds.max(axis=3)[..., None]).astype(int)
#plot_image(X[:1681], y[:1681], preds_t)
