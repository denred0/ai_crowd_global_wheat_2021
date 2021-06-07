import numpy as np
import time
import cv2
import sys
import csv
import torch

from pathlib import Path
import glob
from tqdm import tqdm


# from ensemble_boxes import *


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5/best.pt', force_reload=True)
    model.conf = 0.25  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1)

    test_folder = Path('data/test')
    types = ('*.png')
    COLOR = (255, 0, 0)

    test_files = get_all_files_in_folder(test_folder, types)

    submission = []
    for i, file in tqdm(enumerate(test_files)):
        if i > 0:
        # if file.stem == '00f05dbe4fce3713d1574746de07914dd0f81dea612412ffa09ad73598f85a5c':
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = model(image, size=640)

            results_list = results.pandas().xyxy[0].values.tolist()

            # print('\nresult count:', len(results_list))

            box_string = 'no_box'
            # ensure at least one detection exists
            if len(results_list) > 0:
                box_string = ''
                # loop over the indexes we are keeping
                for res in results_list:
                    # extract the bounding box coordinates
                    (x_min, y_min) = (int(res[0]), int(res[1]))
                    (x_max, y_max) = (int(res[2]), int(res[3]))

                    # color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), COLOR, 2)

                    box_string += str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ';'

            if box_string == 'no_box':
                string_result = str(file.stem) + ',' + str(box_string) + ',' + str(0)
            else:
                string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(0)

            submission.append(string_result)

            if i < 100:
                cv2.imwrite('data/inference_yolov5/' + str(file.stem) + '.png', image)

    with open('data/submission.csv', 'w') as f:
        f.write('image_name,PredString,domain\n')
        for item in submission:
            f.write("%s\n" % item)


def create_csv():
    results = []
    with open('data/submission.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            results.append(row)

    template = []
    with open('data/submission_template.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            template.append(row)

    print(f"results {len(results)}")
    print(f"template {len(template)}")

    nol = []

    for res in results:
        boxes = res[1]
        boxes_list = boxes.split(';')
        for box in boxes_list:
            coords = box.split(' ')
            for coord in coords:
                if '-' in coord:
                    nol.append(coord)
                    # print(coord)

    print(np.unique(sorted(nol)))

    total = []
    for temp in template:
        for res in results:
            if temp[0] == res[0]:
                str_total = str(temp[0]) + ',' + str(temp[1]) + ',' + str(res[1])
                total.append(str_total)

    print(f"total {len(total)}")

    with open('data/sub.csv', 'w') as f:
        for item in total:
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()
    create_csv()
