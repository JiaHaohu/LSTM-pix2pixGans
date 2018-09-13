# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:16:56 2018

@author: hujia
"""

import cv2
import dlib
import os
import numpy as np
import pathlib as plb


detector = dlib.get_frontal_face_detector()
path_68points_model = "D:\\语音图像合成\\dlib_68_dat\\"
pwd = os.getcwd() 
os.chdir(path_68points_model) 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
os.chdir(pwd)

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

    # 如果出现负值，即人脸位于图像框之外的情况，应当忽视图像外的部分，将负值置为0
    if x_min < 0:
        x_min = 0

    if y_min < 0:
        y_min = 0

    print("done")
    return img[y_min-500:y_max+200,x_min-150:x_max+150]
    #return img[y_min:y_max,x_min:x_max]

def cal_face_boundary2(img, shape):
    for index_, pt in enumerate(shape.parts()):
        if index_ == 36:
            x1=pt.x
            y1=pt.y
        elif index_==45:
            x2=pt.x
            y2=pt.y
        else:
            pass
    l=int((x2-x1)/0.35)
    y=int((y1+y2)/2)       

    print("done")
    #return 0
    #return img[y_min-300:y_max+100,x_min-100:x_max+100]
    return img[y-int(l*(2/3)):y+int(l*(2/3)),x1-int(l*0.3):x2+int(l*0.3)]


'''
img = cv2.imread('m093_0123_B.jpg')
#img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
dets = detector(img, 1)
shape = predictor(img,dets[0])
face= cal_face_boundary(img, shape)
face= cv2.resize(face, (70, 40), interpolation=cv2.INTER_LINEAR)
cv2.imshow("img", face)
cv2.imwrite("test.jpg",face)
data=np.array(face[:,:,0])
print(data)
data=data.reshape(1,-1)
cv2.waitKey(0)
print(img.shape)
'''


def cut_photo(picfile):
    img=cv2.imread(picfile)
    #print(img)
    dets=detector(img,1)
    shape=predictor(img,dets[0])
    face=cal_face_boundary2(img,shape)
    os.getcwd()
    os.chdir(picdata)
    #print(face)
    #faceROI = cv2.resize(face, (256,256), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(picfile,face)




dataPath=r"D:\语音图像合成\Face Animation\8-24\pic"
picdata=r"D:\语音图像合成\Face Animation\8-24\pic-test"

Path=plb.Path(dataPath)

if os.path.isdir(picdata)==False:
    os.makedirs(picdata)
    

for actor in Path.iterdir():
    #cut_photo(actor)
    #print(actor)
    temp=str(actor).split("\\")[-1]
    pwd = os.getcwd() 
    os.chdir(dataPath) 
    cut_photo(temp)
    os.chdir(pwd)
 
