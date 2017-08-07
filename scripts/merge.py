import numpy as numpy
import os
import sys
import pickle
import random
import time


if __name__=="__main__":
    net1_dir = './result/vgg_16_0_1/'
    net2_dir = "./result/res10/"
    label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']
    val_path = '/home/xujinchang/share/caffe-center-loss/data/AFEW/video_label/val_log/'
    val_file = os.listdir(val_path)
    
    acc = 0
    count = 0
    for video in val_file:
        f = open(val_path+video,'r')
        
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
        count += 1
        if count==1:
            start_time = time.time()
        for img in imgs:
            score1 = pickle.load(open(net1_dir+video+'/'+img.split("/")[-1], "r"))
            score2 = pickle.load(open(net2_dir+video+'/'+img.split("/")[-1], "r"))
            score = score1+score2
            scores += score
        if int(scores.argmax(axis=0)) == video_label:
            acc += 1
        print "video_name",video,"predict_label",int(scores.argmax(axis=0)),"label",video_label
        print "count: ",count
        f.close()
    print "acc total",acc
    end_time = time.time()
    run_time = end_time - start_time
    print "accuracy: ", float(acc)/(count)
    print "run_time: ",run_time
    print "per_run_time: ",float(run_time)/count
