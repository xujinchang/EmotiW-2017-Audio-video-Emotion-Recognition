import os
fp = open('afew_face_val','r')
fp1 = open('afew_face_val_label','w')
path = '/home/safe_data_dir/EmotiW2017/AEFW/Val/AlignedFaces_LBPTOP_Points_Val/Faces/'
path1 = '/home/safe_data_dir/EmotiW2017/AEFW/Val/'
label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']
file_list = os.listdir(path)


for line in fp.readlines():
	item = line.strip().split('/')
	key = item[-2]
	for label in label_list:
		avi_name = os.listdir(path1+label+'/')
		if key+'.avi' in avi_name:
			fp1.write('/'.join(item)+' '+str(label_list.index(label))+'\n')


fp.close()
fp1.close()
