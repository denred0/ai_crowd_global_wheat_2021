import csv
import cv2

import numpy as np
import shutil

from ensemble_boxes import *
from tqdm import tqdm
from pathlib import Path

from utils import get_all_files_in_folder


def do_wbf():
    dirpath = Path('data/ensemble')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = Path('data/ensemble_init')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    test_files = get_all_files_in_folder(Path('data/test'), ['*.png'])

    yolo_list = []
    with open('data/boxes_conf.csv') as csv_file:
        # with open('data/result_ensemble.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in tqdm(enumerate(csv_reader)):
            yolo_list.append(row)

    efdet_list = []
    with open('data/result_ensemble.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in tqdm(enumerate(csv_reader)):
            efdet_list.append(row)

    print(len(efdet_list))
    print(len(yolo_list))

    image_size_init = 1024
    image_size_inf = 896
    yolo_image_size = 608
    coef_yolo = image_size_init / yolo_image_size
    coef = image_size_init / image_size_inf
    classId = 0

    submission = []
    for yolo in tqdm(yolo_list):
        for efdet in efdet_list:
            yolo_boxes = []
            efdet_boxes = []
            if yolo[0] == efdet[0]:

                boxes = []
                boxes_efdet = []
                if efdet[1] != '' and yolo[1] != '':
                    for box_e in efdet[1].split(';'):
                        box_e_1 = box_e.split(' ')
                        x1 = round(abs(int(box_e_1[0]) / image_size_inf), 2)
                        y1 = round(abs(int(box_e_1[1]) / image_size_inf), 2)
                        x2 = round(int(box_e_1[2]) / image_size_inf, 2)
                        if x2 > 1: x2 = 1
                        y2 = round(int(box_e_1[3]) / image_size_inf, 2)
                        if y2 > 1: y2 = 1
                        efdet_boxes.append([x1, y1, x2, y2])
                        boxes_efdet.append([int(box_e_1[0]), int(box_e_1[1]), int(box_e_1[2]), int(box_e_1[3])])

                    # for file in test_files:
                    #     if file.stem == yolo[0]:
                    #
                    #         image = cv2.imread(str(file), cv2.IMREAD_COLOR)
                    #
                    #         for box in boxes_efdet:
                    #             cv2.rectangle(image, (int(box[0] * coef), int(box[1] * coef)),
                    #                           (int(box[2] * coef), int(box[3] * coef)), (255, 255, 0), 2)
                    #
                    #         cv2.imwrite(str(Path('data/ensemble_init').joinpath(file.name)), image)

                    for box_y in yolo[1].split(';'):
                        box_y_1 = box_y.split(' ')
                        x1 = round(abs(int(box_y_1[0]) / image_size_init), 2)
                        y1 = round(abs(int(box_y_1[1]) / image_size_init), 2)
                        x2 = round((abs(int(box_y_1[2])) + abs(int(box_y_1[0]))) / image_size_init, 2)
                        if x2 > 1: x2 = 1
                        y2 = round((abs(int(box_y_1[3])) + abs(int(box_y_1[1]))) / image_size_init, 2)
                        if y2 > 1: y2 = 1
                        yolo_boxes.append([x1, y1, x2, y2])

                    yolo_conf = [float(x) for x in yolo[2].split(';')]
                    efdet_conf = [float(x) for x in efdet[2].split(';')]

                    yolo_classes = np.ones(len(yolo_conf)).tolist()
                    efdet_classes = np.ones(len(efdet_conf)).tolist()

                    weights = [3, 2]
                    iou_thr = 0.36
                    skip_box_thr = 0.36
                    boxes, scores, labels = weighted_boxes_fusion([efdet_boxes, yolo_boxes], [efdet_conf, yolo_conf],
                                                                  [efdet_classes, yolo_classes], weights=weights,
                                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

                box_str = 'no_box'
                if len(boxes) > 0:
                    box_str = ''
                    for box in boxes:
                        box_str += str(int(box[0] * image_size_inf * coef)) + ' ' + str(
                            int(box[1] * image_size_inf * coef)) + ' ' + str(
                            int(box[2] * image_size_inf * coef)) + ' ' + str(
                            int(box[3] * image_size_inf * coef)) + ';'

                    # for file in test_files:
                    #     if file.stem == yolo[0]:
                    #         # shutil.copy(file, Path('data/ensemble').joinpath(file.name))
                    #
                    #         image = cv2.imread(str(file), cv2.IMREAD_COLOR)
                    #
                    #         for box in boxes:
                    #             cv2.rectangle(image,
                    #                           (
                    #                               int(box[0] * image_size_inf * coef),
                    #                               int(box[1] * image_size_inf * coef)),
                    #                           (
                    #                               int(box[2] * image_size_inf * coef),
                    #                               int(box[3] * image_size_inf * coef)),
                    #                           (255, 255, 0), 2)
                    #
                    #         cv2.imwrite(str(Path('data/ensemble').joinpath(file.name)), image)

                if box_str == 'no_box':
                    sub_str = str(yolo[0]) + ',' + box_str + ',' + str(classId)
                else:
                    sub_str = str(yolo[0]) + ',' + str(box_str[:-1]) + ',' + str(classId)

                submission.append(sub_str)

    return submission


# def check_submission():
#     sumb_list = []
#     with open('data/submission_ensemble.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for i, row in tqdm(enumerate(csv_reader)):
#             sumb_list.append(row)
#
#     sub_good = []
#     with open('data/sub.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for i, row in tqdm(enumerate(csv_reader)):
#             sub_good.append(row)
#
#     test_files = get_all_files_in_folder(Path('data/test'), ['*.png'])
#
#     check = []
#     for file in test_files:
#         exist = False
#         for subm in sumb_list:
#             if subm[0] == file.stem:
#                 exist = True
#
#         check.append([file.stem, exist])

    # for sg in sub_good:

    # for c in check:
    #     if not c[1]:
    #         print(c[0])


def create_csv():
    results = []
    with open('data/submission_ensemble.csv', 'r') as fd:
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

    with open('data/sub_ens.csv', 'w') as f:
        for item in total:
            f.write("%s\n" % item)


def create_scv(subm):
    with open('data/submission_ensemble.csv', 'w') as f:
        f.write('image_name,PredString,domain\n')
        for item in subm:
            f.write("%s\n" % item)


if __name__ == '__main__':
    submission = do_wbf()
    create_scv(submission)

    #
    create_csv()

    # check_submission()
