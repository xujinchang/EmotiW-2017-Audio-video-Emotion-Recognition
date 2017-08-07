import os
import numpy as np
import pickle
label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']
fp = open('/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/single_3D5_test','r')
path = '/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/afew_test_prob/'
video_name = os.listdir(path)

fp1 = open('c3d_lbp_test_128_result.txt','w')
prob_list = []
for line in fp.readlines():
	line = line.strip().split(' ')
	key = line[0].split('/')[-1]
	if key+'_prob' in video_name:
		for item in open(path+key+'_prob','r').readlines():
			item = item.strip().split(' ')
	else:
		item = ['0' for n in range(7)]
	prob_list.append(item)
fp.close()

data = np.array(prob_list,dtype=float)
pickle.dump(data,fp1)
fp1.close()
data1 = pickle.load(open('c3d_lbp_test_128_result.txt','r'))
print len(data1)
