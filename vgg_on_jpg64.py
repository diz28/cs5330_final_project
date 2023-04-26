import io

import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# Di Zhang
# April 22, 2023
# CS5330 - Computer Vision

from resnet_on_jpg64 import get_data
from PIL import Image
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def plot_image(dataset, targets, n_images, truth_or_pred):
    plt.figure()
    for i in range(n_images):
        # set the plot to be 2 by 3
        plt.subplot(int(n_images / 3), 3, i + 1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        plt.imshow(dataset[i][0])
        plt.title("{}: {}".format(truth_or_pred, targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class Transform:
    # initiate the transform
    def __init__(self):
        pass

    # actually implementation of the transform
    def __call__(self, x):
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 1, 0)
        x = torchvision.transforms.functional.center_crop(x, (64, 64))
        x = torchvision.transforms.functional.to_tensor(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    self.samples.append(item)

    def __getitem__(self, index):
        path, target = self.samples[index]

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pil = Image.fromarray(image)

        if self.transform:
            image = self.transform(image_pil)

        image = torch.permute(image, (0, 1, 2))
        return image, target

    def __len__(self):
        return len(self.samples)


def load_images(root_dir, batch_size, shuffle=True):
    file_list = []
    image_num = -1

    to_tensor = transforms.ToTensor()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_list.append((os.path.join(dirpath, filename), os.path.basename(dirpath), image_num))
        image_num = image_num + 1
    if shuffle:
        random.shuffle(file_list)

    num_files = len(file_list)
    num_batches = num_files // batch_size

    crop_size = 64
    transform = transforms.CenterCrop(crop_size)
    dataset = []
    for i in range(num_batches):
        batch_files = file_list[i * batch_size: (i + 1) * batch_size]
        image_list = []
        target_list = []
        for file_path, file_name, image_num in batch_files:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_pil = np.array(image)

            cropped_image = transform(image_pil)
            np_image = np.array(cropped_image)
            image_list.append(np_image)
            target_list.append(image_num)
        tensor_list = torch.tensor(np.array(image_list))
        print(tensor_list.shape)
        tensor_list = tensor_list.permute(0, 3, 1, 2).contiguous()[:, [2, 1, 0], :, :]
        print(tensor_list.shape)
        dataset.append((tensor_list, target_list))

    return dataset


def main():
    # train_dataloader = load_images('./original64JPG/trainset', 32)
    #
    # example_image, example_target = zip(*train_dataset)
    # # plot the first 6 digit images
    # plot_image(example_image, example_target, 9, 'Ground Truth')

    # print(type(example_image))
    # print(example_image)
    # print(type(example_target))
    # print(example_target)
    #

    transform = torchvision.transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset('./original64JPG/trainset', transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    examples = enumerate(train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    print(type(example_data))
    # print(example_data)
    # plot the first 6 digit images
    plot_image(example_data, example_targets, 9, 'Ground Truth')


if __name__ == '__main__':
    main()
