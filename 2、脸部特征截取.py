# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:21:37 2018

@author: hujia
"""

# *_*coding:utf-8 *_*
# author: 许鸿斌

import sys
import cv2
import dlib
import os
import logging
import datetime
import numpy as np

def cal_face_boundary(img, shape):
    for index_, pt in enumerate(shape.parts()):
        if index_ == 0:
            x_min = pt.x
            x_max = pt.x
            y_min = pt.y
            y_max = pt.y
        else:
            if pt.x < x_min:
                x_min = pt.x

            if pt.x > x_max:
                x_max = pt.x

            if pt.y < y_min:
                y_min = pt.y

            if pt.y > y_max:
                y_max = pt.y

    print('x_min:{}'.format(x_min))
    print('x_max:{}'.format(x_max))
    print('y_min:{}'.format(y_min))
    print('y_max:{}'.format(y_max))

    # 如果出现负值，即人脸位于图像框之外的情况，应当忽视图像外的部分，将负值置为0
    if x_min < 0:
        x_min = 0

    if y_min < 0:
        y_min = 0

    print("done")
    return img[y_min-250:y_max+50, x_min-50:x_max+50]

def draw_left_eyebrow(img, shape):
    # 17 - 21
    pt_pos = []
    for index, pt in enumerate(shape.parts()[17:21 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos)-1):
        cv2.line(img, pt_pos[num], pt_pos[num+1], 255, 1)


def draw_right_eyebrow(img, shape):
    # 22 - 26
    pt_pos = []
    for index, pt in enumerate(shape.parts()[22:26 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

def draw_left_eye(img, shape):
    # 36 - 41
    pt_pos = []
    for index, pt in enumerate(shape.parts()[36:41 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

def draw_right_eye(img, shape):
    # 42 - 47
    pt_pos = []
    for index, pt in enumerate(shape.parts()[42:47 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

def draw_nose(img, shape):
    # 27 - 35
    pt_pos = []
    for index, pt in enumerate(shape.parts()[27:35 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[4], 255, 1)
    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)
    cv2.line(img, pt_pos[3], pt_pos[-1], 255, 1)

def draw_mouth(img, shape):
    # 48 - 59
    pt_pos = []
    for index, pt in enumerate(shape.parts()[48:59 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

    # 60 - 67
    pt_pos = []
    for index, pt in enumerate(shape.parts()[60:]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)



def draw_jaw(img, shape):
    # 0 - 16
    pt_pos = []
    for index, pt in enumerate(shape.parts()[0:16 + 1]):
        pt_pos.append((pt.x, pt.y))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)




def draw_photo(picfile):

    
    img = cv2.imread(picfile)
#img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
    dets = detector(img, 1)
    for index, face in enumerate(dets):
        shape = predictor(img, face)
        data=[]
        for index_, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            data.append(pt_pos)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
    
        features = np.zeros(img.shape[0:-1], dtype=np.uint8)
        for index_, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(features, pt_pos, 2, 255, 1)
    draw_left_eyebrow(features, shape)
    draw_right_eyebrow(features, shape)
    draw_left_eye(features, shape)
    draw_right_eye(features, shape)
    draw_nose(features, shape)
    draw_mouth(features, shape)
    draw_jaw(features, shape)
    

    os.getcwd()
    os.chdir(picdata)
    cv2.imwrite(picfile,features)
    print("done")
  



detector = dlib.get_frontal_face_detector()
path_68points_model = "D:\\语音图像合成\\dlib_68_dat\\"
pwd = os.getcwd() 
os.chdir(path_68points_model) 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
os.chdir(pwd)

dataPath=r"D:\语音图像合成\Face Animation\8-24\pic-test"
picdata=r"D:\语音图像合成\Face Animation\8-24\landmarks-test"

Path=plb.Path(dataPath)

if os.path.isdir(picdata)==False:
    os.makedirs(picdata)
    

for actor in Path.iterdir():
    #cut_photo(actor)
    print(actor)
    temp=str(actor).split("\\")[-1]
    pwd = os.getcwd() 
    os.chdir(dataPath) 
    draw_photo(temp)
    os.chdir(pwd)
 
