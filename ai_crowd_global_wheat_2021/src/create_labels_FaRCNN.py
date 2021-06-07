import os
from os import walk
import csv
from pathlib import Path
import shutil
from PIL import Image

from tqdm import tqdm


# ai_crowd_global_wheat_2021
def create_labels_from_csv():
    img_size = 1024

    labels = []
    with open('data/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for i, row in tqdm(enumerate(csv_reader)):
            boxes = row[1].split(";")

            if boxes[0] != 'no_box':

                for box in boxes:
                    box_arr = box.split(" ")

                    x_min = int(box_arr[0])
                    y_min = int(box_arr[1])
                    x_max = int(box_arr[2])
                    y_max = int(box_arr[3])

                    # x_center = (x_min + (x_max - x_min) / 2) / img_size
                    # y_center = (y_min + (y_max - y_min) / 2) / img_size

                    width = (x_max - x_min)
                    height = (y_max - y_min)

                    record = row[0] + ',' + str(img_size) + ',' + str(img_size) + ',' + '\"[' + str(
                        x_min) + ', ' + str(
                        y_min) + ', ' + str(width) + ', ' + str(
                        height) + ']\",usask_1'
                    labels.append(record)

    with open('data/FaRCNN/train.csv', 'w') as f:
        f.write('image_id,width,height,bbox,source\n')
        for item in labels:
            f.write("%s\n" % item)



if __name__ == '__main__':
    create_labels_from_csv()
    # create_dataset()
