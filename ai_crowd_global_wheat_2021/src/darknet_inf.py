import os
import sys
import cv2
import time

from tqdm import tqdm

import pandas as pd

# --------------------------------------------------------------------------------------------------
# Darknet initialization
# --------------------------------------------------------------------------------------------------
sys.path.append('/home/vid/hdd/projects/darknet/')

from darknet import load_network, detect_image

# config_path = "yolo4/exp_11/yolov4-p5-mycustom.cfg"
# weight_path = "yolo4/exp_11/yolov4-p5-mycustom_best.weights"
# meta_path = "yolo4/exp_11/obj.data"

config_path = "yolo4/exp_6/yolov4-obj-mycustom.cfg"
weight_path = "yolo4/exp_6/yolov4-obj-mycustom_best.weights"
meta_path = "yolo4/exp_6/obj.data"

threshold = .38
# hier_threshold=.5
# nms_coeff=.45

net_main, class_names, colors = load_network(config_path, meta_path, weight_path)

# --------------------------------------------------------------------------------------------------
# Read annotations
# --------------------------------------------------------------------------------------------------
csv_file = "data/submission_template.csv"

annotations = pd.read_csv(csv_file)

image_list = annotations["image_name"].values
domain_list = annotations["domain"].values


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def encode_boxes(boxes):
    if len(boxes) > 0:

        boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
        BoxesString = ";".join(boxes)

    else:

        BoxesString = "no_box"

    return BoxesString





def main():
    results = []

    for image, domain in tqdm(zip(image_list, domain_list), total=len(image_list)):

        image_path = "{}/{}.png".format("data/test", image)
        imgArray = cv2.imread(image_path)
        imgArray_to_detect = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)

        startT = time.time()
        detections = detect_image(net_main, class_names, imgArray_to_detect, thresh=threshold)  # Class detection
        detect_time = int(round((time.time() - startT) * 1000))

        # print("\nImage: {}, Time: {} ms".format(image_path, detect_time))

        bboxes = []

        for detection in detections:

            if float(detection[1]) > threshold:

                current_class = detection[0]
                current_thresh = float(detection[1])
                current_coords = [float(x) for x in detection[2]]

                # print("Probability: {:.3f}, Class: {}".format(current_thresh, current_class))

                xmin = float(current_coords[0] - current_coords[2] / 2)
                ymin = float(current_coords[1] - current_coords[3] / 2)
                xmax = float(xmin + current_coords[2])
                ymax = float(ymin + current_coords[3])

                if xmin < 0:
                    xmin = 0

                if xmax > imgArray.shape[1]:
                    xmax = imgArray.shape[1]

                if ymin < 0:
                    ymin = 0

                if ymax > imgArray.shape[0]:
                    ymax = imgArray.shape[0]

                bboxes.append([xmin, ymin, xmax, ymax])

                cv2.rectangle(imgArray, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)

        cv2.imwrite('data/inference_yolov4/' + image + '.png', imgArray)

        PredString = encode_boxes(bboxes)
        results.append([image, PredString, domain.item()])

    results = pd.DataFrame(results, columns=["image_name", "PredString", "domain"])
    results.to_csv("submission_final.csv")

    print("Done!")


if __name__ == "__main__":
    main()
