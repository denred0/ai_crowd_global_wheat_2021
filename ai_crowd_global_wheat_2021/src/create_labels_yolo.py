import os
from os import walk
import csv
from pathlib import Path
import shutil
from PIL import Image

from tqdm import tqdm

# ai_crowd_global_wheat_2021
def create_labels():
    img_size = 1024

    labels = []
    with open('data/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for i, row in tqdm(enumerate(csv_reader)):
            filename = 'data/txts/' + row[0] + '.txt'
            boxes = row[1].split(";")
            domain = row[2]

            if boxes[0] != 'no_box':
                with open(filename, "w") as f:

                    for box in boxes:
                        box_arr = box.split(" ")

                        x_min = int(box_arr[0])
                        y_min = int(box_arr[1])
                        x_max = int(box_arr[2])
                        y_max = int(box_arr[3])

                        x_center = (x_min + (x_max - x_min) / 2) / img_size
                        y_center = (y_min + (y_max - y_min) / 2) / img_size

                        width = (x_max - x_min) / img_size
                        height = (y_max - y_min) / img_size

                        # box_norm = [int(x) / img_size for x in box_arr]
                        domain = 0
                        record = (str(domain) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(
                            width) + ' ' + str(height) + '\n')
                        f.write(record)
                # else:
                #     record = (str(domain) + '\n')
                #     f.write(record)

            labels.append(domain)
    labels = list(set(labels))
    print('Labels: {}'.format(labels))
    print('Labels count: {}'.format(len(labels)))


def create_dataset():
    image_path = 'data/imgs'
    txt_path = 'data/txts'
    _, _, images_list = next(walk(image_path))
    _, _, txt_list = next(walk(txt_path))

    destinationpath = 'data/dataset'

    for txt in tqdm(txt_list):
        for img in images_list:
            if Path(image_path).joinpath(img).stem == Path(txt_path).joinpath(txt).stem:
                shutil.copy(os.path.join(txt_path, txt), os.path.join(destinationpath, txt))
                im = Image.open(Path(image_path).joinpath(img))
                rgb_im = im.convert('RGB')
                rgb_im.save(Path(image_path).joinpath(img).stem + '.jpg')

                # shutil.copy(os.path.join(image_path, img), os.path.join(destinationpath, img))


if __name__ == '__main__':
    create_labels()
    create_dataset()
