# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 00:14:24 2020

@author: bhara
"""
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import torchvision.transforms as T
from flask import Flask,request,jsonify
import io

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

app = Flask(__name__)

@app.route('/fasterrcnn', methods=['POST'])
def hello():
    if request.method == 'POST':
        file = request.files['file']
        imgbyte = file.read()
        img = Image.open(io.BytesIO(imgbyte))
        print(type(img))
        transform = T.Compose([T.Resize(255),T.ToTensor()]) 
        image = transform(img)
        print([image])
        pred = model([image])
        thresh = 0.9
        predicted_labels = [COCO_Class_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
        predicted_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())] 
        predicted_score = list(pred[0]['scores'].detach().numpy())
        predicted_t = [predicted_score.index(x) for x in predicted_score if x > thresh][-1] 
        predicted_boxes = predicted_boxes[:predicted_t+1]
        predicted_labels = predicted_labels[:predicted_t+1]
        predicted_score = predicted_score[:predicted_t+1]
        print(type(predicted_labels), type(predicted_boxes), type(predicted_score))
        return jsonify({'labels': predicted_labels, 'boxes':str(predicted_boxes), 'scores':str(predicted_score)})



COCO_Class_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
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

if __name__ == '__main__':
    app.run()