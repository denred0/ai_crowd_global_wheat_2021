import numpy as np
import time
import cv2
import sys
import csv
import torch
import torchvision

from pathlib import Path
import glob
from tqdm import tqdm
from ensemble_boxes import *
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def main():
    transforms = A.Compose(
        [
            # A.Normalize(mean=mean, std=std),
            # A.Resize(height=img_size, width=img_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(
        # torch.load('data/FaRCNN/fasterrcnn_resnet50_fpn_65_11800.pth', map_location=torch.device('cuda')))
        torch.load('data/FaRCNN/fasterrcnn_resnet50_fpn_41_7600.pth'))
    model.eval()
    model.conf = 0.5
    IMAGE_SIZE = 1024
    model.to(torch.device('cuda:0'))

    test_folder = Path('data/test')
    types = ('*.png')
    COLOR = (255, 0, 0)

    test_files = get_all_files_in_folder(test_folder, types)
    test_files.pop(0)

    LABELS_FILE = 'yolo4/obj.names'
    CONFIG_FILE = 'yolo4/yolov4-obj-mycustom.cfg'
    WEIGHTS_FILE = 'yolo4/yolov4-obj-mycustom_best.weights'
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    CONFIDENCE_THRESHOLD = 0.22

    # valid_dataset = WheatDataset(test_files, 'data/test', get_valid_transform())

    submission = []
    for i, file in tqdm(enumerate(test_files)):
        # if file.stem == '0e782082dd7e10e7a804b97a4efdcea10ee3cf7630956ea064303d78d1052a9f':
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_pred /= 255.0

        image_pred = transforms(image=image_pred)
        # image = image['image'].permute(1,2,0).cpu()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pred = image_pred['image'].unsqueeze(0)
        with torch.no_grad():
            results = model(image_pred.cuda())

        scores_rcnn = results[0]['scores'].detach().cpu().numpy().tolist()
        labels_rcnn = results[0]['labels'].detach().cpu().numpy().tolist()
        boxes_rcnn = results[0]['boxes'].detach().cpu().numpy().astype('int').tolist()

        # yolo
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

        boxes_rcnn_new = []

        for box in boxes_rcnn:
            x1 = round(box[0] / IMAGE_SIZE, 2)
            y1 = round(box[1] / IMAGE_SIZE, 2)

            x2 = round((box[2]) / IMAGE_SIZE, 2)
            y2 = round((box[3]) / IMAGE_SIZE, 2)

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0

            if x1 > 1: x1 = 1
            if y1 > 1: y1 = 1
            if x2 > 1: x2 = 1
            if y2 > 1: y2 = 1

            box_row = [x1, y1, x2, y2]
            boxes_rcnn_new.append(box_row)

        boxes_yolo_new = []

        for box in boxes:
            x1 = round(box[0] / IMAGE_SIZE, 2)
            y1 = round(box[1] / IMAGE_SIZE, 2)

            x2 = round((box[0] + box[2]) / IMAGE_SIZE, 2)
            y2 = round((box[1] + box[3]) / IMAGE_SIZE, 2)

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0

            if x1 > 1: x1 = 1
            if y1 > 1: y1 = 1
            if x2 > 1: x2 = 1
            if y2 > 1: y2 = 1

            box_row = [x1, y1, x2, y2]
            boxes_yolo_new.append(box_row)

        # boxes_new_nms = []

        # for box in boxes:
        #     # x1 = int((box[2] - box[0]) / 2 + box[0])
        #     # y1 = int((box[3] - box[1]) / 2 + box[1])
        #     x1 = box[0]
        #     y1 = box[1]
        #
        #     x2 = (box[2] - box[0])
        #     y2 = (box[3] - box[1])
        #
        #     if x1 < 0: x1 = 0
        #     if y1 < 0: y1 = 0
        #     if x2 < 0: x2 = 0
        #     if y2 < 0: y2 = 0
        #
        #     if x1 > IMAGE_SIZE: x1 = IMAGE_SIZE
        #     if y1 > IMAGE_SIZE: y1 = IMAGE_SIZE
        #     if x2 > IMAGE_SIZE: x2 = IMAGE_SIZE
        #     if y2 > IMAGE_SIZE: y2 = IMAGE_SIZE
        #
        #     box_row = [x1, y1, x2, y2]
        #     boxes_new_nms.append(box_row)
        classIDs = [1] * len(classIDs)

        boxes_wbf, scores, labels = weighted_boxes_fusion([boxes_rcnn_new, boxes_yolo_new], [scores_rcnn, confidences],
                                                          [labels_rcnn, classIDs], weights=[1, 3],
                                                          iou_thr=0.5, skip_box_thr=0.6)
        # idxs = cv2.dnn.NMSBoxes([boxes_rcnn_new], [scores], score_threshold=0.7, nms_threshold=0.7)

        addition_px = 0
        box_string = 'no_box'
        #
        # if len(idxs) > 0:
        #     box_string = ''
        #     # loop over the indexes we are keeping
        #     for i in idxs.flatten():
        #         # extract the bounding box coordinates
        #         (x, y) = (boxes_new_nms[i][0] + addition_px, boxes_new_nms[i][1] + addition_px)
        #         (w, h) = (boxes_new_nms[i][2] - addition_px, boxes_new_nms[i][3] - addition_px)
        #
        #         # color = [int(c) for c in COLORS[classIDs[i]]]
        #         cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)
        #
        #         box_string += str(x) + ' ' + str(y) + ' ' + str(x + w) + ' ' + str(y + h) + ';'

        # ensure at least one detection exists
        if len(boxes_wbf) > 0:
            box_string = ''
        # loop over the indexes we are keeping
        for res in boxes_wbf:
            # extract the bounding box coordinates
            (x_min, y_min) = (int(res[0] * IMAGE_SIZE), int(res[1] * IMAGE_SIZE))
            (x_max, y_max) = (int(res[2] * IMAGE_SIZE), int(res[3] * IMAGE_SIZE))

            # color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), COLOR, 2)

            box_string += str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ';'

        if box_string == 'no_box':
            string_result = str(file.stem) + ',' + str(box_string) + ',' + str(0)
        else:
            string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(0)

        submission.append(string_result)

        cv2.imwrite('data/inference_FaRCNN_yolo/' + str(file.stem) + '.png', image)

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
            # res[0] =
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
