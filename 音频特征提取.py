# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:21:47 2018

@author: hujia
"""

'''
Log_mel倒频谱的一阶差分和二阶差分值组成的128维向量
音频采样率44100 一帧图像对应0.04s 无时窗滑动


'''

import numpy as np
import librosa
from tqdm import tqdm 


def addContext(melSpc, ctxWin): #mel时间信号 ctxwin文本视窗
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]#将filler插入第一行
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    print(cnst)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst
    print("------------")

    melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec

def process_adiuo(path):
    wsize = 0.05  #窗宽
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
z3test=process_adiuo('9-5.aac')
   

'''
下面
'''
'''
sound, sr = librosa.load('3.aac', sr=fs) #时间采样
print("----------------------------")
melFrames = melSpectra(sound, sr, wsize, hsize)

print(melFrames)
print('-----------------------------')
zeroVecD = np.zeros((1, 64), dtype='float16') #一阶差
zeroVecDD = np.zeros((2, 64), dtype='float16') #二阶差



melFrames = np.transpose(melFrames)
print('=================================')
print(melFrames.shape)
melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
print('====================6666===================')
print(melDelta.shape)
print(melDDelta.shape)


features = np.concatenate((melDelta, melDDelta), axis=1)
print(features.shape)
features = addContext(features, 10)
#features = np.reshape(features, (1, features.shape[0], features.shape[1]))
print(features.shape)
'''
np.savetxt('mfcc-9-7.csv',z3test, delimiter = ',')


