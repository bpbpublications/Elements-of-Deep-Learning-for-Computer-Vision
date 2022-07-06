# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:55:23 2020

@author: bhara
"""
import torch
from detecto.core import Model
from detecto import utils, visualize

classlist= ['Chihuahua','golden_retriever']
model = Model(classlist, device = torch.device('cpu'))
model._model.load_state_dict(torch.load(r'C:\Users\bhara\OneDrive\Desktop\Downloads\dogdataset.pth',map_location=torch.device('cpu')))

image = utils.read_image(r'C:\Users\bhara\OneDrive\Desktop\Downloads\dog_dataset\images\n02085620_1321.jpg')  # Helper function to read in images

labels, boxes, scores = model.predict(image)  # Get all predictions on an image
predictions = model.predict_top(image)  # Same as above, but returns only the top predictions

print(labels, boxes, scores)
print(predictions)

visualize.show_labeled_image(image, boxes, labels)