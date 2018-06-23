#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import cv2
import numpy as np

UNIT_SIZE = 229     # the size of image


def pinjie(images, num):
    target = Image.new('RGB', (UNIT_SIZE*5, UNIT_SIZE*2))   # result is 2*5
    leftone = 0
    lefttwo = 0
    rightone = UNIT_SIZE
    righttwo = UNIT_SIZE
    for i in range(len(images)):
        if(i % 2 == 0):
            target.paste(images[i], (leftone, 0, rightone, UNIT_SIZE))
            leftone += UNIT_SIZE   #第一行左上角右移
            rightone += UNIT_SIZE   #右下角右移
        else:
            target.paste(images[i], (lefttwo, UNIT_SIZE, righttwo, UNIT_SIZE*2))
            lefttwo += UNIT_SIZE #第二行左上角右移
            righttwo += UNIT_SIZE #右下角右移
    quality_value = 100
    target.save(path+dirlist[num]+'.jpg', quality = quality_value)


path = './originP'
dirlist = []    # all dir name
for root, dirs, files in os.walk(path):
    for dir in dirs :
        dirlist.append(dir)

num = 0
for dir in dirlist:
    images = [] # images in each folder
    for root, dirs, files in os.walk(path+dir): # traverse each folder
        print (path+dir+'')
        for file in files:
            images.append(Image.open(path+dir+'/'+file))
    pinjie(images,num)
    num +=1
    images = []

    # cv2.putText(img=0, text=1, org=2, fontFace=3, fontScale=4, color=5,thickness=None, lineType=None, bottomLeftOrigin=None)