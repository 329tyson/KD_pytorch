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

logging.basicConfig(
    format='%(message)s',
    filename=str(os.path.dirname(os.path.realpath(__file__)) + '/logs/'+ datetime.datetime.now().strftime('D%d,%H:%M')) + '.log',
    filemode='w',
    level=logging.INFO
)
def timeit(method):
    def timed(*args, **kw):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.info('[{}][{}] spent {} ms'.format(timestamp, "{:>17}".format(method.__name__ + '()'), "{0:.3f}".format((te-ts)*1000)))
        return result
    return timed

class CUBDataset(data.Dataset):

    def __init__(self, csv_file, image_path, trainable = True):
        self.ROOT = os.path.dirname(os.path.realpath(__file__))
        self.CSV_FILE_PATH = os.path.abspath(os.path.join(self.ROOT, csv_file))
        self.IMAGE_PATH = os.path.abspath(os.path.join(self.ROOT, image_path))
        self.IMG_MEAN = np.array([123.68, 116.779, 103.939])
        self.Trainable= trainable
        # self.IMG_MEAN = np.array([103.939, 116.779, 123.68])

        self.load_csv()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.filePath[idx]
        img_label = self.labels[idx]

        if self.Trainable is True:
            image, low_image = self.transform(
                img_path,
                self.Bbox[idx][0],
                self.Bbox[idx][1],
                self.Bbox[idx][2],
                self.Bbox[idx][3]
            )
        else:
            image, low_image = self.generateTest(
                img_path,
                self.Bbox[idx][0],
                self.Bbox[idx][1],
                self.Bbox[idx][2],
                self.Bbox[idx][3]
            )

        return image, low_image, img_label

    @timeit
    def load_csv(self):
        self.filePath= []
        self.labels= []
        self.Bbox= []
        with open(self.CSV_FILE_PATH, 'r') as csvfile:
            cr = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in cr:
                self.filePath.append(self.IMAGE_PATH + '/' + row[0])
                self.Bbox.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
                self.labels.append(int(row[5]))

    def transform(self, img_path, x_, y_, w_, h_):

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # b, g, r = cv2.split(image)
        # image = cv2.merge([r,g,b])
        image = image[y_:y_+h_, x_:x_+w_]
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        image = image - self.IMG_MEAN

        # Random flip
        if random.random() > 0.5 :
            image = cv2.flip(image,1)

        y_crop = random.random() * 30
        y_crop = int(y_crop)

        x_crop = random.random() * 30
        x_crop = int(x_crop)

        image = image[y_crop:y_crop + 227,x_crop:x_crop +227]

        # image = transforms.ToTensor()(image)

        # Low res
        low_image = cv2.resize(image, (25,25), interpolation=cv2.INTER_CUBIC)
        low_image = cv2.resize(low_image, (227,227), interpolation=cv2.INTER_CUBIC)

        image = transforms.ToTensor()(image)
        low_image = transforms.ToTensor()(low_image)

        return image, low_image

    def generateTest(self, img_path, x_, y_, w_, h_):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = image[y_:y_+h_, x_:x_+w_]
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        image = image - self.IMG_MEAN

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

        # print('image 1 shape : ',img1.shape)
        # print('image 6 shape : ',img6.shape)
        image = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]

        low_image = [cv2.resize(img, (25, 25), interpolation=cv2.INTER_CUBIC) for img in image]
        low_image = [cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC) for img in low_image]
        low_image = [transforms.ToTensor()(img) for img in low_image]
        low_image = np.stack(low_image, axis=0)

        image = [transforms.ToTensor()(img) for img in image]
        image = np.stack(image, axis = 0)

        return image, low_image

