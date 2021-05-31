import torch
import cv2
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5/last.pt', force_reload=True)
model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)


# # Image
img = cv2.imread('data/yolo5_data/example.png')[:, :, ::-1]

# Inference
results = model(img)
# results.show()
p = results.pandas().xyxy[0]



