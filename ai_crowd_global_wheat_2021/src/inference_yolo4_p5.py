import numpy as np
import time
import cv2
import sys
import csv
import torch
import shutil

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
    # dirpath = Path('data/inference_yolov4')
    # if dirpath.exists() and dirpath.is_dir():
    #     shutil.rmtree(dirpath)
    # Path(dirpath).mkdir(parents=True, exist_ok=True)

    # yolo4-tiny
    LABELS_FILE = 'yolo4/exp_11/obj.names'
    CONFIG_FILE = 'yolo4/exp_11/yolov4-p5-mycustom.cfg'
    WEIGHTS_FILE = 'yolo4/exp_11/yolov4-p5-mycustom_best.weights'

    # # classes labels for png classes representation
    # classes_dict = {}
    # with open('yolo4/classes_map.txt') as f:
    #     for line in f:
    #         (key, val) = line.split()
    #         classes_dict[int(key)] = val

    CONFIDENCE_THRESHOLD = 0.4  # 0.26
    inference_image_size = 896  # 736
    NMS_THR = 0.4 # 0.4

    # LABELS = open(LABELS_FILE).read().strip().split("\n")

    # colors for bounding boxes
    np.random.seed(4)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    COLOR = (255, 0, 0)

    # get model
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    test_folder = Path('data/test')
    types = ['*.png']

    test_files = get_all_files_in_folder(test_folder, types)

    submission = []
    boxes_conf = []
    for ind, file in tqdm(enumerate(test_files), total=len(test_files)):
        if file.stem == '0b4d11b3c59fd74a095090da71abf87808c58d57a84639b99e8b2eafdd29cd7f':
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)

            (H, W) = image.shape[:2]
            # size = (W, H)
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (inference_image_size, inference_image_size),
                                         swapRB=True, crop=False)
            net.setInput(blob)

            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            str_boxes = ''

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > CONFIDENCE_THRESHOLD:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            box_str = ''
            for box in boxes:
                box_str += str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ';'

            score_str = ''
            for sc in confidences:
                score_str += str(sc) + ';'

            str_boxes = str(file.stem) + ',' + box_str[:-1] + ',' + score_str[:-1]
            boxes_conf.append(str_boxes)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONFIDENCE_THRESHOLD,
                                    nms_threshold=NMS_THR)

            box_string = 'no_box'
            # ensure at least one detection exists
            if len(idxs) > 0:
                box_string = ''
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)

                    box_string += str(x) + ' ' + str(y) + ' ' + str(x + w) + ' ' + str(y + h) + ';'

            if box_string == 'no_box':
                string_result = str(file.stem) + ',' + str(box_string) + ',' + str(classID)
            else:
                string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(classID)

            submission.append(string_result)

            # if ind < 100:
            cv2.imwrite('data/inference_yolov4/' + str(file.stem) + '.png', image)

    with open('data/submission.csv', 'w') as f:
        f.write('image_name,PredString,domain\n')
        for item in submission:
            f.write("%s\n" % item)

    with open('data/boxes_conf.csv', 'w') as f:
        for item in boxes_conf:
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
