# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 23:37:07 2020

@author: bhara
"""
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'