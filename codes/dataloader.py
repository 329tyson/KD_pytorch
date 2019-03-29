import cv2
import random
import numpy as np
import csv
from torch.utils import data
from torchvision import transforms
import time
import datetime
import logging
import os
import scipy.io

class Dataset(data.Dataset):

    def __init__(self, dataset, annotation_path, image_path, low_ratio, data_type, ten_crop, KD_flag = False, image_norm = False):
        self.ROOT = os.path.dirname(os.path.realpath(__file__))
        self.DATASET = dataset
        self.ANNOTATION = annotation_path
        self.IMAGE_PATH = image_path
        self.IMG_MEAN = np.array([123.68, 116.779, 103.939])
        self.TYPE = data_type
        self.RATIO = low_ratio
        self.ten_crop = ten_crop
        self.isKD = KD_flag
        self.image_norm = image_norm
        self.normalise = transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.IMG_MEAN = np.array([103.939, 116.779, 123.68])

        if self.DATASET.lower() == 'cub':
            self.load_csv()
        elif self.DATASET.lower() == 'stanford':
            self.load_mat()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.filePath[idx]
        img_label = self.labels[idx]

        if self.isKD is True:
            image, low_image= self.transform(
                img_path,
                self.Bbox[idx][0],
                self.Bbox[idx][1],
                self.Bbox[idx][2],
                self.Bbox[idx][3]
            )
            # return image, low_image, img_label
            return image, low_image, img_label, img_path

        if self.ten_crop is False:
            image= self.transform(
                img_path,
                self.Bbox[idx][0],
                self.Bbox[idx][1],
                self.Bbox[idx][2],
                self.Bbox[idx][3]
            )
        else:
            image= self.generateTest(
                img_path,
                self.Bbox[idx][0],
                self.Bbox[idx][1],
                self.Bbox[idx][2],
                self.Bbox[idx][3]
            )

        return image, img_label
        # return image, img_label, img_path

    def load_mat(self):
        self.filePath= []
        self.labels= []
        self.Bbox= []
        matfile = scipy.io.loadmat(self.ANNOTATION)
        for row in matfile['annotations'][0]:
            if self.TYPE == 'Train' and int(row[6]) == 0:
                self.filePath.append(self.IMAGE_PATH + '/' + str(row[0][0]))
                self.Bbox.append([int(row[1][0]), int(row[2][0]), int(row[3][0]), int(row[4][0])])
                self.labels.append(int(row[5][0]))
            elif self.TYPE == 'Validation' and int(row[6]) == 1:
                self.filePath.append(self.IMAGE_PATH + '/' + str(row[0][0]))
                self.Bbox.append([int(row[1][0]), int(row[2][0]), int(row[3][0]), int(row[4][0])])
                self.labels.append(int(row[5][0]))

    def load_csv(self):
        self.filePath= []
        self.labels= []
        self.Bbox= []
        with open(self.ANNOTATION, 'r') as csvfile:
            cr = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in cr:
                self.filePath.append(self.IMAGE_PATH + '/' + row[0])
                self.Bbox.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
                self.labels.append(int(row[5]))

    def transform(self, img_path, x_, y_, w_, h_):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # b, g, r = cv2.split(image)
        # image = cv2.merge([r,g,b])
        if self.DATASET.lower() == 'cub':
            image = image[y_:y_+h_, x_:x_+w_]
        else:
            image = image[y_:h_, x_:w_]
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        # image = image - self.IMG_MEAN

        # Random flip
        if random.random() > 0.5 :
            image = cv2.flip(image,1)

        y_crop = random.random() * 30
        y_crop = int(y_crop)

        x_crop = random.random() * 30
        x_crop = int(x_crop)

        image = image[y_crop:y_crop + 227,x_crop:x_crop +227]

        if self.isKD is True :
            low_image = cv2.resize(image, (self.RATIO,self.RATIO), interpolation=cv2.INTER_CUBIC)
            low_image = cv2.resize(low_image, (227,227), interpolation=cv2.INTER_CUBIC)
            low_image = transforms.ToTensor()(low_image)
            image = transforms.ToTensor()(image)

            if self.image_norm:
                low_image = self.normalise(low_image)
                image = self.normalise(image)
            return image, low_image


        # Low res if needed
        if self.RATIO != 0:
            low_image = cv2.resize(image, (self.RATIO,self.RATIO), interpolation=cv2.INTER_CUBIC)
            low_image = cv2.resize(low_image, (227,227), interpolation=cv2.INTER_CUBIC)
            low_image = transforms.ToTensor()(low_image)
            if self.image_norm:
                low_image = self.normalise(low_image)

            return low_image

        image = transforms.ToTensor()(image)
        if self.image_norm:
            image = self.normalise(image)
        return image

    def generateTest(self, img_path, x_, y_, w_, h_):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.DATASET.lower() == 'cub':
            image = image[y_:y_+h_, x_:x_+w_]
        else:
            image = image[y_:h_, x_:w_]
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        # image = image - self.IMG_MEAN

        img1 = image[0:227, 0:227]
        img2 = image[29:256, 0:227]
        img3 = image[0:227, 29:256]
        img4 = image[29:256, 29:256]
        img5 = image[14:241, 14:241]

        img6 = cv2.flip(img1,1)
        img7 = cv2.flip(img2,1)
        img8 = cv2.flip(img3,1)
        img9 = cv2.flip(img4,1)
        img10 = cv2.flip(img5,1)

        image = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]

        if self.RATIO != 0:
            low_image = [cv2.resize(img, (self.RATIO,self.RATIO), interpolation=cv2.INTER_CUBIC) for img in image]
            low_image = [cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC) for img in low_image]

            low_image = [transforms.ToTensor()(img) for img in low_image]
            if self.image_norm:
                low_image = [self.normalise(img) for img in low_image]
            low_image = np.stack(low_image, axis=0)

            return low_image

        image = [transforms.ToTensor()(img) for img in image]
        if self.image_norm:
            image = [self.normalise(img) for img in image]
        image = np.stack(image, axis = 0)

        return image
