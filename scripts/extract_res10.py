#coding:utf-8
import numpy as np
import time
import os
import json
import sys
import socket
import copy
import math
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0,'./python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

#MODEL_DEF='/home/xujinchang/caffe_liveness/caffe-master/models/bvlc_alexnet/blur_deploy.prototxt'
#MODEL_DEF = '/home/xujinchang/caffe-blur-pose/models/vgg/deploy_vgg_emo19.prototxt'
#MODEL_DEF = '/home/xujinchang/caffe-blur-pose/models/vgg/deploy_vgg_fer.prototxt'
#MODEL_PATH = '/home/zhangjiangqi/project/pretrained/resnet/101-caffe/ResNet-101-model.caffemodel'
#MODEL_PATH = '../models/vgg_19_finetue_fer2013_iter_30000.caffemodel'

MODEL_DEF = '/home/xujinchang/share/caffe-center-loss/emotiw/afew/resnet/depoly/deploy_res10.prototxt'
MODEL_PATH = './models_xu/res10_afewface_video_0_iter_5400.caffemodel'
#MODEL_DEF = '/home/malilei/share/emotiw/resnet/deploy/deploy_res26_afew_0.prototxt'
# MODEL_DEF = '/home/malilei/share/emotiw/resnet/deploy/deploy_res34_afew_0.prototxt'
# MODEL_PATH = '/home/malilei/share/emotiw/resnet/models/res34_feature1_0/_iter_2000.caffemodel'
#MODEL_PATH = '/home/malilei/share/emotiw/resnet/models/res26_feature1_0/_iter_3000.caffemodel'
#MODEL_PATH = '/home/malilei/share/emotiw/resnet/models/res34_feature1_0/_iter_2000.caffemodel'
mean = np.array((104.0, 117.0, 123.0), dtype=np.float32)
SIZE = 150

def predict(the_net,image):
  inputs = []
  if not os.path.exists(image):
    raise Exception("Image path not exist")
    return
  try:
    tmp_input = cv2.imread(image)
    tmp_input = cv2.resize(tmp_input,(SIZE,SIZE))
    tmp_input = tmp_input[11:11+128,11:11+128]
    tmp_input = np.subtract(tmp_input,mean)
    tmp_input = tmp_input.transpose((2, 0, 1))
    tmp_input = np.require(tmp_input, dtype=np.float32)
  except Exception as e:
    #raise Exception("Image damaged or illegal file format")
    return None
  the_net.blobs['data'].reshape(1, *tmp_input.shape)
  the_net.reshape()
  the_net.blobs['data'].data[...] = tmp_input
  the_net.forward()
  scores = copy.deepcopy(the_net.blobs['feature'].data)
  return scores

if __name__=="__main__":
  # img_dir = "/localSSD/hezhiqun/image_quality/pick_result/blur/"
  #img_dir = "/home/xujinchang/share/AGG/data/test1/"
  #f = open("/home/xujinchang/share/AGG/Liveness/detection/sort_testcodafinal","rb")
  f = open("./lstm_afew_face/afew_face_lstm_train.txt","rb")
  fp = open("./lstm_afew_face/res10_lbp_face_train.fea","w")
  fp2 = open("./lstm_afew_face/res10_lbp_face_train_label.fea","w")
  net = caffe.Net(MODEL_DEF, MODEL_PATH, caffe.TEST)

  start_time = time.time()
  X_features=[]
  y_label=[]
  count = 0
  for line in f.readlines():
    line = line.strip().split(" ")
    print line[0]
    fea = predict(net,line[0])
    fea = list(np.reshape(fea, (fea.shape[1], fea.shape[0])))
    feature = np.require(fea)# (4096,1)
    X_features.append(feature)
    count = count + 1
    if count % 16 == 0:
      y_label.append(line[1])

  print len(X_features)
  print len(y_label)
  for item in X_features:
      for idx in range(item.shape[0]):
          fp.write(str(item[idx][0])+' ')
      fp.write('\n')
  #print X_features[10]
  for item in y_label:
      fp2.write(str(item)+'\n')
  #print X_features[-1]

  f.close()
  fp.close()
  fp2.close()
  #print X_features[0]
  end_time = time.time()
  forward_time = end_time - start_time
  print "forward_time: ",forward_time

