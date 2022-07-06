# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 02:06:48 2020

@author: bhara
"""

from PIL import Image, ImageEnhance 
import webcolors  
img = Image.open(r'C:\Users\bhara\OneDrive\Desktop\Book\images\Dress_jpg.jpg') 

#Get Colour Name 
def colour(colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - colour[0]) ** 2
        gd = (g_c - colour[1]) ** 2
        bd = (b_c - colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]
    
#Iterate over the pixels to find the pixel bands
pixellist= []
width, height = img.size
for x in range(width):
    for y in range(height):
        [r,g,b] = img.getpixel((x,y))
        pixellist.append(colour_name((r,g,b)))
        
#Get all unique colours and count repetitions across image
fullist = {}
allcolours = []
for i in set(pixellist):    
    closest_name = colour(i)
    allcolours.append(closest_name)

freq = {} 
for item in allcolours: 
    if (item in freq): 
        freq[item] += 1
    else: 
        freq[item] = 1

#Sort the list to give us the most occuring colour        
freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])}
for key, value in freq.items(): 
    print ("% s : % d"%(key, value))