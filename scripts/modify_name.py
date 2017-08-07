import os

filename = '/home/xujinchang/share/caffe-center-loss/data/AFEW/C3D_label/AFEW_train_label'

fp = open(filename,'r')
fp_log = open('less16_val.txt','w')
fp_w = open('c3d_modify_val_all.txt','w')
fp1 = open('c3d_train_shiji.txt','w')
name_dict = {}
label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']
for line in fp.readlines():
	ori_name_path = line.strip().split(' ')[0]
	image_name=ori_name_path.split('/')[-1]
	video_name = '/'.join(ori_name_path.split('/')[:-1])
	name_dict.setdefault(video_name,[]).append(image_name)

for key in name_dict:
	image_list = name_dict[key]
	num = len(image_list)
	sort_image_list = sorted(image_list)
	if num < 16:
		print >>fp_log,key,num
		continue
	else:
		count = 1
		for item in sort_image_list:
			# if count < 10: 
			# 	os.system("mv {name} {name1}".format(name=key+'/'+item,name1=key+'/'+'00000'+str(count)+'.jpg'))
			# 	fp_w.write(key+'/'+'00000'+str(count)+'.jpg'+'\n')
			# if 10<=count and count<=99: 
			# 	os.system("mv {name} {name1}".format(name=key+'/'+item,name1=key+'/'+'0000'+str(count)+'.jpg'))
			# 	fp_w.write(key+'/'+'0000'+str(count)+'.jpg'+'\n')
			# if 100<=count:
			# 	os.system("mv {name} {name1}".format(name=key+'/'+item,name1=key+'/'+'000'+str(count)+'.jpg'))
			# 	fp_w.write(key+'/'+'000'+str(count)+'.jpg'+'\n')
			if count == 1:
				label = label_list.index(key.split('/')[-2])
				fp1.write(key+'/'+' '+'1'+' '+str(label)+'\n')
			count = count + 1

fp.close()
fp_log.close()
fp1.close()
fp_w.close()