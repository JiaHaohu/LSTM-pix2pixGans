# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 09:19:18 2018

@author: hujia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:01:14 2018

@author: hujia
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
import librosa
import os
from imutils import video
import time
'''
声音处理
'''


def addContext(melSpc, ctxWin): #mel时间信号 ctxwin文本视窗
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]#将filler插入第一行
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    #print(cnst)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst
    #print("------------")

    melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec

def process_adiuo(path):
    wsize = 0.02  #窗宽
    hsize = 0.05
    fs = 44100 #每秒时间采样频率
    sound, sr = librosa.load(path, sr=fs) #时间采样
    melFrames = melSpectra(sound, sr, wsize, hsize)
    zeroVecD = np.zeros((1, 64), dtype='float16') #一阶差
    zeroVecDD = np.zeros((2, 64), dtype='float16') #二阶差
    melFrames = np.transpose(melFrames)
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
    features = np.concatenate((melDelta, melDDelta), axis=1)
    features = addContext(features, 10)
    features = np.reshape(features, (features.shape[0], features.shape[1]))
    
    return features

#mfcc=process_adiuo('9-5.aac')

def load_graph(frozen_graph_filename):
    graph=tf.Graph()
    with graph.as_default():
        od_graph_def=tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename,'rb') as fid:
            serialized_graph=fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')
    return  graph

def run1(mfcc):
    graph=load_graph("lstm-9-13-70.pb")
    input_=graph.get_tensor_by_name('input_x:0')
    keep_prob=graph.get_tensor_by_name('keep:0')
    output_=graph.get_tensor_by_name('LSTM/prd:0')
    sess=tf.Session(graph=graph)
    pred=sess.run(output_,feed_dict={input_:mfcc,keep_prob:0.8})
    sess.close()
    return pred
def run2(src):
    #tf.reset_default_graph()
    graph=load_graph("frozen_model.pb")
    input_image=graph.get_tensor_by_name('input_image:0')
    output_image=graph.get_tensor_by_name('generator/output_image:0')
    sess=tf.Session(graph=graph)
    #src=cv2.imread("922.png")
    #src=cv2.resize(src,(256,256), interpolation=cv2.INTER_LINEAR)
    generated_image=sess.run(output_image,feed_dict={input_image:src})
    image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
    return image_bgr
   #cv2.imshow("test",image)
   

'''
画脸
'''
def draw_left_eyebrow(img, shape):
    # 17 - 21
    pt_pos = []
    for index, pt in enumerate(shape[17:21 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos)-1):
        cv2.line(img, pt_pos[num], pt_pos[num+1], 255, 1)


def draw_right_eyebrow(img, shape):
    # 22 - 26
    pt_pos = []
    for index, pt in enumerate(shape[22:26 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

def draw_left_eye(img, shape):
    # 36 - 41
    pt_pos = []
    for index, pt in enumerate(shape[36:41 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

def draw_right_eye(img, shape):
    # 42 - 47
    pt_pos = []
    for index, pt in enumerate(shape[42:47 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

def draw_nose(img, shape):
    # 27 - 35
    pt_pos = []
    for index, pt in enumerate(shape[27:35 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[4], 255, 1)
    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)
    cv2.line(img, pt_pos[3], pt_pos[-1], 255, 1)

def draw_mouth(img, shape):
    # 48 - 59
    pt_pos = []
    for index, pt in enumerate(shape[48:59 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)

    # 60 - 67
    pt_pos = []
    for index, pt in enumerate(shape[60:]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)

    cv2.line(img, pt_pos[0], pt_pos[-1], 255, 1)



def draw_jaw(img, shape):
    # 0 - 16
    pt_pos = []
    for index, pt in enumerate(shape[0:16 + 1]):
        pt_pos.append((pt[0], pt[1]))

    for num in range(len(pt_pos) - 1):
        cv2.line(img, pt_pos[num], pt_pos[num + 1], 255, 1)




def draw_photo(shape):

    
 
    
    features = np.zeros((256,256,3), dtype=np.uint8)

    for face in shape:
        cv2.circle(features,(face[0],face[1]), 2, 255, 1)
        
    draw_left_eyebrow(features, shape)
    draw_right_eyebrow(features, shape)
    draw_left_eye(features, shape)
    draw_right_eye(features, shape)
    draw_nose(features, shape)
    draw_mouth(features, shape)
    draw_jaw(features, shape)
    
    return features
    #cv2.imshow('111',features)
    #cv2.waitKey(0)
    #print("done")
'''
扩充 眼睛鼻子眉毛
输入 音频数量len(mfcc)
'''
def process_eye(length):
    temp=np.loadtxt('landmarks.csv',delimiter = ',').astype(np.float32)
    temp_len=temp.shape[0]
    rat=length//temp_len
    print(int(rat))
    if rat>0:
        res=length%temp_len
        y_org=temp
        for i in range(int(rat)-1):
            y_org=np.concatenate((y_org,temp),axis=0)
        y_org=np.concatenate((y_org,temp[:res]))
        return y_org
    elif rat==0:
        y_org=temp[:length]
        return y_org

if __name__ == '__main__':
    
    
    mfcc=process_adiuo('9-5.aac')
    #mfcc=np.ones((20000,1408))
    y_org=process_eye(len(mfcc))
    tf.reset_default_graph()
    #mfcc=mfcc.astype(tf.float32)
    
    pred=(run1(mfcc))
    
    y_org[:,-38:]=pred[:,:38]*256
    y_org[:,:32]=pred[:,-32:]*256
    landmark=y_org
    tf.reset_default_graph()
    graph=load_graph("frozen_model.pb")
    input_image=graph.get_tensor_by_name('input_image:0')
    output_image=graph.get_tensor_by_name('generator/output_image:0')
    sess=tf.Session(graph=graph)
    #a=input()
    #tf.reset_default_graph()
    fps = video.FPS().start()
    for i in range(len(mfcc)):
        #landmark=(run1(mfcc[i].reshape(-1,1408))*256).astype(int)
        src=draw_photo(landmark[i].reshape(68,2))
        #cv2.imshow('test',src)
        
        #image=run2(src)
        #cv2.imshow('test',image)
        #cv2.waitKey(0)
        print(i)
        #fps.update()
    #src=cv2.imread("922.png")
    #src=cv2.resize(src,(256,256), interpolation=cv2.INTER_LINEAR)
        generated_image=sess.run(output_image,feed_dict={input_image:src})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('test',image_bgr)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.time(0.02)
    
    fps.stop()
    cv2.destroyAllWindows()
        #run2(src)
