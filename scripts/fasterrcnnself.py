# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 00:24:12 2020

@author: bhara
"""

import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import torchvision.transforms as T
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_Class_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'guinea pig', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

img = Image.open(r'C:\Users\bhara\OneDrive\Desktop\Book\images\cat.jpg')
transform = T.Compose([T.ToTensor()]) 
image = transform(img)
pred = model([image])
thresh = 0.5
# print(pred)
predicted_labels = [COCO_Class_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
predicted_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())] 
predicted_score = list(pred[0]['scores'].detach().numpy())
predicted_t = [predicted_score.index(x) for x in predicted_score if x > thresh][-1] 
predicted_boxes = predicted_boxes[:predicted_t+1]
predicted_labels = predicted_labels[:predicted_t+1]
predicted_score = predicted_score[predicted_t+1]
print(type(predicted_labels), type(predicted_boxes), type(list(predicted_score)))

plt.imshow(img)
for i in range(len(predicted_boxes)):
    print(predicted_boxes[i])
    print(predicted_labels[i])
    rect = patches.Rectangle((predicted_boxes[i][0],predicted_boxes[i][1]),predicted_boxes[i][2],predicted_boxes[i][3],linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(predicted_boxes[i][0],predicted_boxes[i][1], '{}'.format(predicted_labels[i]), color = 'blue', size=10)
plt.axis('off')
plt.savefig(r'C:\Users\bhara\OneDrive\Desktop\Book\images\cat_predicted.jpg',bbox_inches='tight')
# boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
# img = cv2.imread(img_path) # Read image with cv2
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
# for i in range(len(boxes)):
#   cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
#   cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
# plt.figure(figsize=(20,30)) # display the output image
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()




# pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
# pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
# pred_score = list(pred[0]['scores'].detach().numpy())
# pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
# pred_boxes = pred_boxes[:pred_t+1]
# pred_class = pred_class[:pred_t+1]