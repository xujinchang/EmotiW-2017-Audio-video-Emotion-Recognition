import os
filename = '/local/home/share/xujinchang/project/C3D/C3D-v1.0/examples/c3d_train_ucf101/c3d_val_shiji.txt'
fp = open(filename,'r')
fp1 = open('lstm_val.txt','w')
for line in fp.readlines():
	line = line.strip().split(' ')
	key = line[0]
	label = line[2]
	image_list = os.listdir(key)
	image_list = sorted(image_list)
	count = 1
	for image in image_list:
		if image[1] != '0' :continue
		if count > 16:break
		fp1.write(key+image+' '+label+'\n')
		count += 1
fp.close()
fp1.close()