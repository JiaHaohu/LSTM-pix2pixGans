# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:04:06 2018

@author: hujia
"""

import os
import cv2
import numpy as np
import math
import sys
import dlib
from skimage import io
import faceProcess as face
import matplotlib.pyplot as plt
import pathlib as plb

path_68points_model = "D:\\语音图像合成\\dlib_68_dat\\"   
vedioPath="E:\人脸合成动画\speech_data\SAVEE"
picPath="D:\语音图像合成\Face Animation\dataset\pic1www"
AAMPath="D:\语音图像合成\Face Animation\dataset\AAM-19"


def shape_to_np(shape):
    coords=np.zeros((68,2),dtype=int)
    
    for i in range(0,68):
        coords[i]=(int(shape.part(i).x),int(shape.part(i).y))
    return coords

def get_68points_model(pictureFile,path_68points_model):
    img = io.imread(pictureFile)
    pwd = os.getcwd() 
    os.chdir(path_68points_model)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    os.chdir(pwd)
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    shape = predictor(img, dets[0])
    coords=shape_to_np(shape)
    coords=process_coord(coords)
    return coords

def process_coord(coords):
    w=h=600
    #eyecornerDst=[80,,(np.int(0.7*w),np.int(h/3))]
    eyecornerDst=[(80,112),(175,110)]
    eyecornerSrc  = [ coords[36,:], coords[45,:]]
    tform =face.similarityTransform(eyecornerSrc, eyecornerDst)
    points2 = np.reshape(coords, (68,1,2));        
 
    points = cv2.transform(points2, tform);
 
    points = np.float32(np.reshape(points, (68, 2)))
    return points
    
    
    
'''
处理过程
1.脸部对齐(眼睛) 仿射变化
2.归一化
'''
landmarks=np.zeros((3762,136))
path=r"pic-resize"

for i in range(1,3763):
    data=get_68points_model(path+'/%d.png'%i,path_68points_model)
    data=data.reshape(1,136)
    landmarks[i-1,:]=data
    print(i)
np.savetxt('landmarks-9-int.csv',landmarks, delimiter = ',')
'''
data=get_68points_model(path,path_68points_model)
plt.figure(figsize=(9,10))
ax = plt.gca()  
    
#    ax.xaxis.set_ticks_position('top')
    
ax.invert_yaxis()  #y轴反向
plt.xlim(0,1)
plt.ylim(0,1)
ax.invert_yaxis()  #y轴反向
plt.scatter(data[:,0],data[:,1])
'''