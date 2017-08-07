filename = '/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_train_ucf101/afew_face/single_afew_val'
fp = open(filename,'r')

fp1 = open('afew_face_lstm_single_val.txt','w')

for line in fp.readlines():
	line = line.strip().split(' ')
	path = line[0]
	frame_id = int(line[1])
	label = line[2]
	for i in range(frame_id,frame_id+16):
		fp1.write(path+str(i).zfill(6)+'.jpg'+' '+label+'\n')
fp.close()
fp1.close()