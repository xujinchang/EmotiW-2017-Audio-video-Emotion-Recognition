#coding:utf-8
import numpy as np
import time
import os
import sys
sys.path.insert(0,'./python')
import json
import sys
import socket
import logging
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import caffe
import random
import copy
import pickle

caffe.set_mode_gpu()
caffe.set_device(3)

#MODEL_DEF = "../deploy/VGG_fc6_deploy.prototxt"
#MODEL_PATH = "../model/VGG/dan_vgg_iter_50000.caffemodel"
#MODEL_DEF2 = "../deploy/Alexnet_deploy.prototxt"
#MODEL_PATH2 = "../model/Alexnet/dan_alexnet_iter_14000.caffemodel"
#MODEL_DEF = './deploy_task/depoly_DAN50.prototxt'
#MODEL_PATH = '/home/xujinchang/share/project/transfor_learning/transfer-caffe/model_xu/DAN_res50_iter_100000.caffemodel'
#MODEL_DEF2 = './deploy_task/depoly_JAN50.prototxt'
#MODEL_PATH2 = '/home/xujinchang/share/project/transfor_learning/transfer-caffe/model_xu/JAN_res50_iter_100000.caffemodel'

MODEL_DEF = '/home/xujinchang/share/caffe-center-loss/emotiw/afew/resnet/depoly/deploy_res10.prototxt'
MODEL_PATH = './models_xu/res10_afewface_video_single_iter_6400.caffemodel'
#MODEL_PATH = './models_xu/res18_face_fer_iter_36000.caffemodel'
#MODEL_PATH = './models_xu/res10_face_video_0_iter_30000.caffemodel'
#MODEL_PATH = './models_xu/res10_face_fer_iter_50000.caffemodel'
#MODEL_PATH = './good_model/vgg_face16_finetue2w_fer2013_iter_20000.caffemodel'
SIZE = 150
CROP_SIZE = 128

mean = np.array((104.0, 117.0, 123.0), dtype=np.float32)
def predict(image,the_net):
    inputs = []
    try:
        tmp_input = image
        tmp_input = cv2.resize(tmp_input,(SIZE,SIZE))
        tmp_input = tmp_input[11:11+128,11:11+128];
        tmp_input = np.subtract(tmp_input,mean)
        tmp_input = tmp_input.transpose((2, 0, 1))
        tmp_input = np.require(tmp_input, dtype=np.float32)
    except Exception as e:
        raise Exception("Image damaged or illegal file format")
        return
    the_net.blobs['data'].reshape(1, *tmp_input.shape)
    the_net.reshape()
    the_net.blobs['data'].data[...] = tmp_input
    the_net.forward()
    scores = the_net.blobs['prob'].data[0]
    return copy.deepcopy(scores)

if __name__=="__main__":
    #f = open("./fer2013/fer2013_valid","rb")
    #f_w = open("pred_fer2013valid.txt","wb")
    #f = open("./data/AFEW/AFEW_val_label","rb")
    f = open("afew_face_lstm_single_val.txt","rb")
    f_w = open("pred_afew.txt","wb")
    net = caffe.Net(MODEL_DEF, MODEL_PATH, caffe.TEST)
    img_label = dict()
    for line in f.readlines():
        line = line.strip().split(" ")
        img_label[line[0]] = line[1]
    count = 0
    acc = 0
    imgs = img_label.keys()
    indexs = range(len(imgs))
    random.shuffle(indexs)
    imgs = [imgs[i] for i in indexs]
    for img in imgs:
        count += 1
        if count==1:
            start_time = time.time()
        cv_img = cv2.imread(img)
        scores = predict(cv_img,net)
        if int(scores.argmax(axis=0)) == int(img_label[img]):
            acc += 1
        f_w.write(img+" "+str(int(scores.argmax(axis=0)))+"\n")
        print "count: ",count
    f.close()
    f_w.close()
    end_time = time.time()
    run_time = end_time - start_time
    print "accuracy: ", float(acc)/len(imgs)
    print "run_time: ",run_time
    print "per_run_time: ",float(run_time)/count
