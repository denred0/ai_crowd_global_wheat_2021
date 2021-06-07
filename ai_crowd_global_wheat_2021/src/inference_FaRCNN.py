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


# class WheatDataset(Dataset):
#
#     def __init__(self, images_list, image_dir, transforms=None):
#         super().__init__()
#
#         self.image_ids = images_list
#         self.image_dir = image_dir
#         self.transforms = transforms
#
#     def __getitem__(self, index: int):
#         image_id = self.image_ids[index]
#         # records = self.df[self.df['image_id'] == image_id]
#
#         image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         image /= 255.0
#
#         # boxes = records[['x', 'y', 'w', 'h']].values
#         # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
#         # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
#         #
#         # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # area = torch.as_tensor(area, dtype=torch.float32)
#
#         # there is only one class
#         # labels = torch.ones((records.shape[0],), dtype=torch.int64)
#
#         # suppose all instances are not crowd
#         # iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
#
#         # target = {}
#         # target['boxes'] = boxes
#         # target['labels'] = labels
#         # # target['masks'] = None
#         # target['image_id'] = torch.tensor([index])
#         # target['area'] = area
#         # target['iscrowd'] = iscrowd
#
#         if self.transforms:
#             image = self.transforms(image=image)
#             # image = sample['image']
#
#             # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
#
#         return image, 0, image_id
#
#     def __len__(self) -> int:
#         return self.image_ids.shape[0]


# def get_valid_transform():
#     return A.Compose(
#         [
#             # A.Normalize(mean=mean, std=std),
#             # A.Resize(height=img_size, width=img_size, p=1.0),
#             ToTensorV2(p=1.0),
#         ],
#         p=1.0,
#         bbox_params=A.BboxParams(
#             format='pascal_voc',
#             min_area=0,
#             min_visibility=0,
#             label_fields=['labels']
#         )
#     )


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
        torch.load('data/FaRCNN/fasterrcnn_resnet50_fpn_65_11800.pth', map_location=torch.device('cpu')))
    model.eval()
    model.conf = 0.2
    IMAGE_SIZE = 1024
    # cpu_device = torch.device("cpu")

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5/best.pt', force_reload=True)
    # model.conf = 0.25  # confidence threshold (0-1)
    # model.iou = 0.45  # NMS IoU threshold (0-1)

    test_folder = Path('data/test')
    types = ('*.png')
    COLOR = (255, 0, 0)

    test_files = get_all_files_in_folder(test_folder, types)
    test_files.pop(0)

    # valid_dataset = WheatDataset(test_files, 'data/test', get_valid_transform())

    submission = []
    for i, file in tqdm(enumerate(test_files)):
        # if file.stem == '00727db2685f6f7f49f5589e602b1e29d3cbd0642df86e86d9a5a968570d0bf0':
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_pred /= 255.0

        image_pred = transforms(image=image_pred)
        # image = image['image'].permute(1,2,0).cpu()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pred = image_pred['image'].unsqueeze(0)
        with torch.no_grad():
            results = model(image_pred)

        scores = results[0]['scores'].detach().numpy().tolist()
        labels = results[0]['labels'].detach().numpy().tolist()
        boxes = results[0]['boxes'].detach().numpy().astype('int').tolist()

        # boxes_new = []
        #
        # for box in boxes:
        #     x1 = box[0] / IMAGE_SIZE
        #     y1 = box[1] / IMAGE_SIZE
        #
        #     x2 = (box[2]) / IMAGE_SIZE
        #     y2 = (box[3]) / IMAGE_SIZE
        #
        #     if x1 < 0: x1 = 0
        #     if y1 < 0: y1 = 0
        #     if x2 < 0: x2 = 0
        #     if y2 < 0: y2 = 0
        #
        #     if x1 > 1: x1 = 1
        #     if y1 > 1: y1 = 1
        #     if x2 > 1: x2 = 1
        #     if y2 > 1: y2 = 1
        #
        #     box_row = [x1, y1, x2, y2]
        #     boxes_new.append(box_row)

        boxes_new_nms = []
        for box in boxes:
            # x1 = int((box[2] - box[0]) / 2 + box[0])
            # y1 = int((box[3] - box[1]) / 2 + box[1])
            x1 = box[0]
            y1 = box[1]

            x2 = (box[2] - box[0])
            y2 = (box[3] - box[1])

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0

            if x1 > IMAGE_SIZE: x1 = IMAGE_SIZE
            if y1 > IMAGE_SIZE: y1 = IMAGE_SIZE
            if x2 > IMAGE_SIZE: x2 = IMAGE_SIZE
            if y2 > IMAGE_SIZE: y2 = IMAGE_SIZE

            box_row = [x1, y1, x2, y2]
            boxes_new_nms.append(box_row)

        # boxes_wbf, scores, labels = weighted_boxes_fusion([boxes_new], [scores], [labels], weights=None,
        #                                                   iou_thr=0.5, skip_box_thr=0.2)
        idxs = cv2.dnn.NMSBoxes(boxes_new_nms, scores, score_threshold=0.6, nms_threshold=0.5)

        if len(idxs) > 0:
            box_string = ''
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes_new_nms[i][0], boxes_new_nms[i][1])
                (w, h) = (boxes_new_nms[i][2], boxes_new_nms[i][3])

                # color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)

                box_string += str(x) + ' ' + str(y) + ' ' + str(x + w) + ' ' + str(y + h) + ';'

        # box_string = 'no_box'
        # # ensure at least one detection exists
        # if len(boxes_wbf) > 0:
        #     box_string = ''
        # # loop over the indexes we are keeping
        # for res in boxes_wbf:
        #     # extract the bounding box coordinates
        #     (x_min, y_min) = (int(res[0] * IMAGE_SIZE), int(res[1] * IMAGE_SIZE))
        #     (x_max, y_max) = (int(res[2] * IMAGE_SIZE), int(res[3] * IMAGE_SIZE))
        #
        #     # color = [int(c) for c in COLORS[classIDs[i]]]
        #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), COLOR, 2)
        #
        #     box_string += str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ';'

        if box_string == 'no_box':
            string_result = str(file.stem) + ',' + str(box_string) + ',' + str(0)
        else:
            string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(0)

        submission.append(string_result)

        cv2.imwrite('data/inference_FaRCNN/' + str(file.stem) + '.png', image)

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
            if temp[0] == res[0][1:]:
                str_total = str(temp[0]) + ',' + str(temp[1]) + ',' + str(res[1])
                total.append(str_total)

    print(f"total {len(total)}")

    with open('data/sub.csv', 'w') as f:
        for item in total:
            f.write("%s\n" % item)


if __name__ == '__main__':
    # main()
    create_csv()
