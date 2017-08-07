import os
import numpy as np
import pickle

fp = open('/home/xujinchang/share/caffe-center-loss/sklearn_model/our_vgg_testfc6_prob','r')
data = pickle.load(fp)


# fp1 = open('/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_train_ucf101/our_face_val_shiji.txt','r')
# fp2 = open('/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_train_ucf101/afew_face/single_afew_val','r')
fp1 = open('/home/xujinchang/share/caffe-center-loss/lstm_afew_face/3D4_no0_log')
fp2 = open('/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_3D5_test')
fp3 = open('map_to_ourtrainval_testfc6','w')
our_face_list=[]
for line in fp1.readlines():
	line = line.strip().split(' ')
	key = line[0].split('/')[-2]
	our_face_list.append(key)
fp1.close()
print our_face_list
prob_list = []
for line in fp2.readlines():
	line = line.strip().split(' ')
	key = line[0].split('/')[-2]
	if key in our_face_list:
		print key,our_face_list.index(key)
		item = data[our_face_list.index(key)]
	else:
		item = ['0' for n in range(7)]
	prob_list.append(item)
fp2.close()

map_prob = np.array(prob_list,dtype=float)
pickle.dump(map_prob,fp3)
fp.close()
fp3.close()
data1 = pickle.load(open('map_to_ourtrainval_testfc6','r'))
print len(data1)
