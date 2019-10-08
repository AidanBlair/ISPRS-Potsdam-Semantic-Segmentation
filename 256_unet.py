# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:24:41 2019

@author: Aidan
"""

import os
import random

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from skimage.io import imread
import tifffile as tiff

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plt.style.use("ggplot")

BATCH_SIZE = 16


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = sorted(os.listdir(img_folder))
    o = sorted(os.listdir(mask_folder))

    while True:
        img = np.zeros((batch_size, 256, 256, 3)).astype("float")
        mask = np.zeros((batch_size, 256, 256, 6)).astype("bool")

        for i in range(c, c+batch_size):
            train_img = imread(img_folder+"/"+n[i])/255.
            img[i-c] = train_img
            train_mask = tiff.imread(mask_folder+"/"+o[i])
            mask[i-c] = train_mask

        c += batch_size
        if (c+batch_size >= len(os.listdir(img_folder))):
            c = 0

        yield img, mask


train_frame_path = "data3/train_data"
train_mask_path = "labels3/train_labels"

val_frame_path = "data3/val_data"
val_mask_path = "labels3/val_labels"

test_frame_path = "data3/test_data"
test_mask_path = "labels3/test_labels"

train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
test_gen = data_gen(test_frame_path, test_mask_path, batch_size=BATCH_SIZE)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layers
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


num_classes = 6
im_height = 256
im_width = 256


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3,
                      batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3,
                      batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3,
                      batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3,
                      batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3,
                      batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2),
                         padding="same")(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3,
                      batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2),
                         padding="same")(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3,
                      batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2),
                         padding="same")(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3,
                      batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2),
                         padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3,
                      batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input(shape=(im_height, im_width, 3), name="img")
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

print(model.summary())

model.compile(optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["acc"])

#model.load_weights("new-segmentation-model.h5")

callbacks = [EarlyStopping(patience=10, verbose=True),
             ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001,
                               verbose=True),
<<<<<<< HEAD
             ModelCheckpoint("256_unet_checkpoint_50.h5", save_best_only=True, save_weights_only=True)]
=======
             ModelCheckpoint("256_unet_50.h5", save_best_only=True,
                             save_weights_only=True)]
>>>>>>> 43891338e1a5e1968b3d6667af2b46d09e6cc347

num_training_samples = 38880
num_validation_samples = 5556
num_test_samples = 6348
num_epochs = 50
results = model.fit_generator(generator=train_gen,
                              steps_per_epoch=num_training_samples//BATCH_SIZE,
                              epochs=num_epochs, callbacks=callbacks,
                              validation_data=val_gen,
                              validation_steps=num_validation_samples//BATCH_SIZE,
                              verbose=2)

model.save_weights("256_unet_50.h5")

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
plt.savefig("256_unet_lossplot_50.png", bbox_inches="tight")
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
plt.savefig("256_unet_accplot_50.png", bbox_inches="tight")
plt.show()

# Load best model
model.save_weights("256_unet_50.h5")

test_loss, test_acc = model.evaluate_generator(generator=test_gen,
                                               steps=num_test_samples,
                                               verbose=2)
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
y_pred = model.predict(X_test)
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
    X_ = imread("data3/train_data/train_data{}.png".format(i)).reshape((1, 256, 256, 3))/255.
    X_ = tiff.imread("labels3/train_labels/train_labels{}.tif".format(i)).reshape((1, 256, 256, 3))
    X[i-0*2116] = X_
    y[i-0*2116] = y_

preds_train = model.predict(X, verbose=True)
preds_train_t = (preds_train == preds_train.max(axis=3)[..., None]).astype(int)

def plot_labels(X, y, preds):
    width = 45
    height = 45
    image_array = np.zeros((6000, 6000, 3), dtype=np.int)
    labels_array = np.zeros((6000, 6000, 6), dtype=np.bool)
    preds_array = np.zeros((6000, 6000, 6))
    for i in range(width):
        for j in range(height):
            image_array[i*(128):(i+1)*(128)+128, j*(128):(j+1)*(128)+128, :] = X[i*46+j]
            labels_array[i*(128):(i+1)*(128)+128, j*(128):(j+1)*(128)+128, :] = y[i*46+j]
            preds_array[i*(128):(i+1)*(128)+128, j*(128):(j+1)*(128)+128, :] = preds[i*46+j]
        image_array[i*128:(i*128+256), 5744:6000, :] = X[i*46+45]
        labels_array[i*128:(i*128+256), 5744:6000, :] = y[i*46+45]
        preds_array[i*128:(i*128+256), 5744:6000, :] = preds[i*46+45]
    for j in range(height):
        image_array[5744:6000, j*128:(j*128+256), :] = X[j+2070]
        labels_array[5744:6000, j*128:(j*128+256), :] = y[j+2070]
        preds_array[5744:6000, j*128:(j*128+256), :] = preds[j+2070]
    image_array[5744:6000, 5744:6000, :] = X[-1]
    labels_array[5744:6000, 5744:6000, :] = y[-1]
    preds_array[5744:6000, 5744:6000, :] = preds[-1]

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image_array, interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("256_unet_full_image1.png", bbox_inches="tight")
    plt.show()

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
    plt.savefig("256_unet_full_labels1.png", bbox_inches="tight")
    plt.show()

plot_labels(X, y, preds_train_t)

