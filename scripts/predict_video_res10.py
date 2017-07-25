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
caffe.set_device(0)


MODEL_DEF = '/home/xujinchang/share/caffe-center-loss/emotiw/resnet/depoly/deploy_res10.prototxt'
#MODEL_PATH = './models_xu/vgg_face19_finetue2w_fer2013_iter_20000.caffemodel'
MODEL_PATH = './models_xu/res10_face_video_0_iter_18000.caffemodel'

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
    label_list = ['Angry','Disgust','Fear','Neutral','Surprise','Sad','Happy']
    val_path = '/home/xujinchang/share/caffe-center-loss/data/AFEW/video_label/val_log/'
    val_file = os.listdir(val_path)
    net1_dir = './result/res10/'
    acc = 0
    count = 0
    net = caffe.Net(MODEL_DEF, MODEL_PATH, caffe.TEST)
    for video in val_file:
        f = open(val_path+video,'r')
        f_w = open('./result/res10_log/'+video+'_pre',"wb")
        
        img_label = dict()
        video_label = label_list.index(video.split('_')[0])
        
        for line in f.readlines():
            line = line.strip().split(" ")
            img_label[line[0]] = line[1]
        imgs = img_label.keys()
        indexs = range(len(imgs))
        random.shuffle(indexs)
        imgs = [imgs[i] for i in indexs]
        scores = 0
        if not os.path.exists(net1_dir+video):
            os.makedirs(net1_dir+video)
        count += 1
        if count==1:
            start_time = time.time()
        for img in imgs:
            cv_img = cv2.imread(img)
            score = predict(cv_img,net)
            scores += score
            pickle.dump(score, open(net1_dir+video+'/'+img.split("/")[-1], "w"))
        if int(scores.argmax(axis=0)) == video_label:
            acc += 1
        print "video_name",video,"predict_label",int(scores.argmax(axis=0)),"label",video_label
        print "count: ",count
        f_w.write(video+" "+str(int(scores.argmax(axis=0)))+"\n")
        f.close()
        f_w.close()
    print "acc total",acc
    end_time = time.time()
    run_time = end_time - start_time
    print "accuracy: ", float(acc)/(count)
    print "run_time: ",run_time
    print "per_run_time: ",float(run_time)/count

