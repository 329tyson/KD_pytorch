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

    def __init__(self, csv_file, image_path):
        self.ROOT = os.path.dirname(os.path.realpath(__file__))
        self.CSV_FILE_PATH = os.path.abspath(os.path.join(self.ROOT, csv_file))
        self.IMAGE_PATH = os.path.abspath(os.path.join(self.ROOT, image_path))
        self.IMG_MEAN = np.array([123.68, 116.779, 103.939])

        self.load_csv()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.filePath[idx]
        img_label = self.labels[idx]

        # print(img_path)
        image = self.transform(
            img_path,
            self.Bbox[idx][0],
            self.Bbox[idx][1],
            self.Bbox[idx][2],
            self.Bbox[idx][3]
        )

        return image, img_label

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
        # image = image - self.IMG_MEAN


        y_crop = random.random() * 30
        y_crop = int(y_crop)

        x_crop = random.random() * 30
        x_crop = int(x_crop)

        image = image[0:227,0:227]
        normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229])
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         # std=[0.229, 0.224, 0.225])

        image = transforms.ToTensor()(image)
        image = normalize(image)

        # image = image[0:227,0:227]

        # image = cv2.resize(image, (50,50), interpolation=cv2.INTER_CUBIC)
        # image = cv2.resize(image, (227,227), interpolation=cv2.INTER_CUBIC)

        return image

