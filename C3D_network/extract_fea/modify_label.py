import os
label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']
fp = open('/local/home/share/xujinchang/project/C3D/C3D-v1.0/examples/c3d_train_ucf101/afew_face/3D_Val_log','r')
path = '/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/afew_face_train/'
fp1 = open('afew_val_128_result.txt','r')
dict = {}
for line in fp.readlines():
	line = line.strip().split(' ')
	key = line[0].split('/')[-2]
	value = line[2]
	dict[key] = value
count = 0
for line in fp1.readlines():
	line = line.strip().split(' ')
	key = line[0].split('/')[-2]
	if int(dict[key]) == int(line[1]):
		count += 1

print float(count)/363

fp.close()
fp1.close()
