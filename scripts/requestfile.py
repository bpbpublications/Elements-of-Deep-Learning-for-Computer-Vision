# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 00:31:01 2020

@author: bhara
"""

import requests

resp = requests.post("http://127.0.0.1:5000/fasterrcnn",
                     files={"file": open(r'C:\Users\bhara\OneDrive\Desktop\Book\images\cat.jpg','rb')})

print(resp.content)