# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:41:59 2019

@author: Aidan
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, MaxPool2d, Module, \
    ModuleList, Sequential
from torch.nn.functional import relu, softmax, cross_entropy
from torch.optim import Adam
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 1
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def conv(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
    c = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
               padding=padding)
    if batch_norm:
        bn = BatchNorm2d(out_channels)
        return Sequential(c, bn)
    return c


class DownConv(Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv_in = conv(self.in_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = relu(self.conv_in(x))
        x = relu(self.conv_out(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = ConvTranspose2d(self.in_channels, self.out_channels,
                                      kernel_size=2, stride=2)

        self.conv_in = conv(2 * self.out_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up, output_size=from_down.size())
        x = torch.cat((from_up, from_down), 1)
        x = relu(self.conv_in(x))
        x = relu(self.conv_out(x))
        return x


class SegmentationUNet(Module):
    def __init__(self, num_classes, device, in_channels=3, depth=5,
                 start_filts=64):
        super(SegmentationUNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.device = device

        self.down_convs = []
        self.up_convs = []

        outs = 0
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = conv(outs, self.num_classes, kernel_size=1,
                               padding=0, batch_norm=False)

        self.down_convs = ModuleList(self.down_convs)
        self.up_convs = ModuleList(self.up_convs)

    def forward(self, x):
        x = x.to(self.device)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x


class TverskyCrossEntropyDiceWeightedLoss(Module):
    def __init__(self, num_classes, device):
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def tversky_loss(self, pred, target, alpha=0.5, beta=0.5):
        target_oh = torch.eye(self.num_classes)[target.squeeze(1)]
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        probs = softmax(pred, dim=1)
        target_oh = target_oh.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        inter = torch.sum(probs * target_oh, dims)
        fps = torch.sum(probs * (1 - target_oh), dims)
        fns = torch.sum((1 - probs) * target_oh, dims)
        t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
        return 1 - t

    def class_dice(self, pred, target, epsilon=1e-6):
        pred_class = torch.argmax(pred, dim=1)
        dice = np.ones(self.num_classes)
        for c in range(self.num_classes):
            p = (pred_class == c)
            t = (target == c)
            inter = (p * t).sum().float()
            union = p.sum() + t.sum() + epsilon
            d = 2 * inter / union
            dice[c] = 1 - d
        return torch.from_numpy(dice).float()

    def forward(self, pred, target, cross_entropy_weight=0.5,
                tversky_weight=0.5):
        if cross_entropy_weight + tversky_weight != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should '
                             'sum to 1')
        ce = cross_entropy(
                pred, target, weight=self.class_dice(
                        pred, target).to(self.device))
        tv = self.tversky_loss(pred, target)
        loss = (cross_entropy_weight * ce) + (tversky_weight * tv)
        return loss


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size, num_classes, device):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(
                self.image_paths[idx]).resize((self.size, self.size),
                                              resample=Image.BILINEAR))
        image = image / 255
        mask = np.array(
                tiff.imread(str(self.mask_paths[idx])), dtype='int')
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float().to(self.device)
        mask = np.moveaxis(mask, -1, 0)
        mask = torch.from_numpy(mask).long().to(self.device)
        return image, mask


class SegmentationAgent:
    def __init__(self, val_percentage, test_num, num_classes,
                 batch_size, img_size, data_path, shuffle_data,
                 learning_rate, device):
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_list, self.masks_list = self.load_data(data_path, "val")
        train_split, val_split, test_split = self.make_splits(
            val_percentage, test_num, shuffle_data)
        self.test_split = self.load_data(data_path, "test")
        self.train_loader = self.get_dataloader(train_split)
        self.validation_loader = self.get_dataloader(val_split)
        self.test_loader = self.get_dataloader(self.test_split)
        self.model = SegmentationUNet(self.num_classes, self.device)
        self.criterion = TverskyCrossEntropyDiceWeightedLoss(self.num_classes,
                                                             self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def load_data(self, path, division):
        images_list = list(path.glob('data2/'+division+'_data/*.png'))
        masks_list = list(path.glob('labels2/'+division+'_labels/*.tif'))
        if len(images_list) != len(masks_list):
            raise ValueError('Invalid data loaded')
        images_list = np.array(images_list)
        masks_list = np.array(masks_list)
        return images_list, masks_list

    def make_splits(self, val_percentage, test_num, shuffle=True):
        if shuffle:
            shuffle_idx = np.random.permutation(range(len(self.images_list)))
            self.images_list = self.images_list[shuffle_idx]
            self.masks_list = self.masks_list[shuffle_idx]

        val_num = len(self.images_list) - int(val_percentage * len(
                self.images_list))
        train_images = self.images_list[:val_num]
        train_masks = self.masks_list[:val_num]

        validation_images = self.images_list[val_num:-test_num]
        validation_masks = self.masks_list[val_num:-test_num]

        test_images = self.images_list[-test_num:]
        test_masks = self.masks_list[-test_num:]

        return (train_images, train_masks), \
               (validation_images, validation_masks), \
               (test_images, test_masks)

    def get_dataloader(self, split):
        return DataLoader(SegmentationDataset(split[0], split[1],
                                              self.img_size, self.num_classes,
                                              self.device),
                          self.batch_size, shuffle=False)


classes = ("cars", "buildings", "low_vegetation", "trees", "impervious",
           "clutter")

agent = SegmentationAgent(0.2, 1, 6, 1, 160,
                          Path("C:/Users/Aidan/ISPRS Potsdam"), False, 0.001,
                          'cpu')

agent.model.load_state_dict(torch.load("torch_unet.h5"))
'''
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(agent.train_loader, 0):
        inputs, labels = data

        agent.optimizer.zero_grad()

        outputs = agent.model(inputs)
        loss = agent.criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        agent.optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print("[%d, %5d] loss: %3.f" %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
'''
print("Finished Training")
torch.save(agent.model.state_dict(), "torch_unet.h5")

dataiter = iter(agent.train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images.cpu()))
#print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

outputs = agent.model(images)

_, predicted = torch.max(outputs, 1)

#print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))
'''
correct = 0
total = 0
with torch.no_grad():
    for data in agent.train_loader:
        images, labels = data
        labels = torch.argmax(labels, 1).reshape((-1, 160, 160))
        outputs = agent.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) * labels.size(1) * labels.size(2)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on the 10000 test images: %d %%" %
      (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in agent.train_loader:
        images, labels = data
        labels = torch.argmax(labels, 1)
        outputs = agent.model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze().reshape((-1, 160, 160))
        for i in range(c.size(0)):
            label = labels[i].reshape(
                    (labels[i].size(-1)*labels[i].size(-2), -1)).squeeze()
            c_ = c[i].reshape((c[i].size(-1)*c.size(-2), -1)).squeeze()
            for j in range(len(label)):
                class_correct[label[j].item()] += c_[j].item()
                class_total[label[j].item()] += 1

for i in range(6):
    print("Accuracy of %5s : %2d %%" %
          (classes[i], 100 * class_correct[i] / class_total[i]))

'''
def plot_image(X, y, preds):
    X = np.moveaxis(X, 1, -1)
    y = np.moveaxis(y, 1, -1)
    preds = np.moveaxis(preds, 1, -1)
    width = 41
    height = 41
    overlap = 14
    image_array = np.zeros((width*160-(width-1)*overlap, height*160-(height-1)*overlap, 3), dtype=np.int)
    labels_array = np.zeros((width*160-(width-1)*overlap, height*160-(height-1)*overlap, 6), dtype=np.bool)
    preds_array = np.zeros((width*160-(width-1)*overlap, height*160-(height-1)*overlap, 6))
    for i in range(width):
        for j in range(height):
            image_array[i*(160-overlap):(i+1)*(160-overlap)+overlap, j*(160-overlap):(j+1)*(160-overlap)+overlap, :] = X[i*width+j]
            labels_array[i*(160-overlap):(i+1)*(160-overlap)+overlap, j*(160-overlap):(j+1)*(160-overlap)+overlap, :] = y[i*width+j]
            preds_array[i*(160-overlap):i*(160-overlap)+overlap, j*(160-overlap):(j+1)*(160-overlap)+overlap, :] += preds[i*width+j, :overlap, :, :]
            preds_array[i*(160-overlap):(i+1)*(160-overlap)+overlap, j*(160-overlap):j*(160-overlap)+overlap, :] += preds[i*width+j, :, :overlap, :]
            preds_array[i*(160-overlap)+overlap:(i+1)*(160-overlap)+overlap, j*(160-overlap)+overlap:(j+1)*(160-overlap)+overlap, :] += preds[i*width+j, overlap:, overlap:, :]

    # cars = yellow
    true_cars_overlay = (labels_array[..., 0] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay*1), axis=2)
    # buildings = blue
    true_buildings_overlay = (labels_array[..., 1] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay*1), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (labels_array[..., 2] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay*1), axis=2)
    # trees = green
    true_trees_overlay = (labels_array[..., 3] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay*1), axis=2)
    # impervious = white
    true_impervious_overlay = (labels_array[..., 4] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay*1), axis=2)
    # clutter = red
    true_clutter_overlay = (labels_array[..., 5] > 0).reshape((labels_array.shape[0], labels_array.shape[1], 1))
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay*1), axis=2)

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    ax[0].imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax[0].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax[0].grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    print("Ding")

    # cars = yellow
    true_cars_overlay = (preds_array[..., 0] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_cars_overlay_rgba = np.concatenate((true_cars_overlay, true_cars_overlay, np.zeros(true_cars_overlay.shape), true_cars_overlay*1), axis=2)
    # buildings = blue
    true_buildings_overlay = (preds_array[..., 1] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_buildings_overlay_rgba = np.concatenate((np.zeros(true_buildings_overlay.shape), np.zeros(true_buildings_overlay.shape), true_buildings_overlay, true_buildings_overlay*1), axis=2)
    # low_vegetation = cyan
    true_low_vegetation_overlay = (preds_array[..., 2] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_low_vegetation_overlay_rgba = np.concatenate((np.zeros(true_low_vegetation_overlay.shape), true_low_vegetation_overlay, true_low_vegetation_overlay, true_low_vegetation_overlay*1), axis=2)
    # trees = green
    true_trees_overlay = (preds_array[..., 3] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_trees_overlay_rgba = np.concatenate((np.zeros(true_trees_overlay.shape), true_trees_overlay, np.zeros(true_trees_overlay.shape), true_trees_overlay*1), axis=2)
    # impervious = white
    true_impervious_overlay = (preds_array[..., 4] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_impervious_overlay_rgba = np.concatenate((true_impervious_overlay, true_impervious_overlay, true_impervious_overlay, true_impervious_overlay*1), axis=2)
    # clutter = red
    true_clutter_overlay = (preds_array[..., 5] > 0).reshape((preds_array.shape[0], preds_array.shape[1], 1))
    true_clutter_overlay_rgba = np.concatenate((true_clutter_overlay, np.zeros(true_clutter_overlay.shape), np.zeros(true_clutter_overlay.shape), true_clutter_overlay*1), axis=2)

    ax[1].imshow(true_cars_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_buildings_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_low_vegetation_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_trees_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_impervious_overlay_rgba, interpolation="bilinear")
    ax[1].imshow(true_clutter_overlay_rgba, interpolation="bilinear")
    ax[1].grid(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()

'''
#agent.model.to("cuda")
X_image = np.zeros((100, 3, 160, 160))
y_image = np.zeros((100, 6, 160, 160))
imageiter = iter(agent.test_loader)
for (i, data) in enumerate(imageiter):
    if i < 100:
        im, lab = data
        X_image[i] = im
        y_image[i] = lab
    else:
        break
with torch.no_grad():
    preds_image = agent.model(torch.from_numpy(X_image).float()).numpy()
'''
X = torch.from_numpy(np.moveaxis(np.load("data/final_train_data0.npy"), -1, 1))
print("Ding")
y = torch.from_numpy(np.moveaxis(np.load("labels/final_train_labels0.npy"), -1, 1))
print("Dong")
#preds = agent.model(X.float())

plot_image(X.numpy(), y.numpy(), y.numpy())
