import numpy as np
import time
import cv2
import sys
import csv

from pathlib import Path
import glob
from tqdm import tqdm
from ensemble_boxes import *


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def main():
    # yolo4-tiny
    LABELS_FILE = 'yolo4/obj.names'
    CONFIG_FILE = 'yolo4/yolov4-obj-mycustom.cfg'
    WEIGHTS_FILE = 'yolo4/yolov4-obj-mycustom_best.weights'

    # # classes labels for png classes representation
    # classes_dict = {}
    # with open('yolo4/classes_map.txt') as f:
    #     for line in f:
    #         (key, val) = line.split()
    #         classes_dict[int(key)] = val

    CONFIDENCE_THRESHOLD = 0.20

    LABELS = open(LABELS_FILE).read().strip().split("\n")

    # colors for bounding boxes
    np.random.seed(4)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    COLOR = (255, 0, 0)

    # get model
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    test_folder = Path('data/test')
    types = ('*.png')

    test_files = get_all_files_in_folder(test_folder, types)

    submission = []
    for i, file in tqdm(enumerate(test_files)):
        if i > 0:
        # if file.stem == '1aeba2df1a75fc2c195295d0868eb8c2bf2a3495c0061e18d7fdfe82f734de82':
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)

            (H, W) = image.shape[:2]
            # size = (W, H)
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 1024),
                                         swapRB=True, crop=False)
            net.setInput(blob)

            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

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
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                                    CONFIDENCE_THRESHOLD)


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

            cv2.imwrite('data/inference_result/' + str(file.stem) + '.png', image)

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
    # create_csv()
