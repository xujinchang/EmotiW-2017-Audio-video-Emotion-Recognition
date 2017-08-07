import os

path = '/home/tmp_data_dir/emotj2017/AFEW_FACE_VAL/Faces/'
video_name = os.listdir(path)

fp_log = open('less16_afew_val.txt','w')
fp_w = open('c3d_afew_face_val.txt','w')
#fp1 = open('c3d_train_shiji.txt','w')

for item in video_name:
	image_list = os.listdir(path+item)
	num = len(image_list)
	sort_image_list = sorted(image_list)
	if num < 16:
		print >>fp_log,item,num
		continue	
	else:
		count = 1
		for image in sort_image_list:
			if count < 10: 
				os.system("mv {name} {name1}".format(name=path+item+'/'+image,name1=path+item+'/'+'00000'+str(count)+'.jpg'))
				fp_w.write(path+item+'/'+'00000'+str(count)+'.jpg'+'\n')
			if 10<=count and count<=99: 
				os.system("mv {name} {name1}".format(name=path+item+'/'+image,name1=path+item+'/'+'0000'+str(count)+'.jpg'))
				fp_w.write(path+item+'/'+'0000'+str(count)+'.jpg'+'\n')
			if 100<=count:
				os.system("mv {name} {name1}".format(name=path+item+'/'+image,name1=path+item+'/'+'000'+str(count)+'.jpg'))
				fp_w.write(path+item+'/'+'000'+str(count)+'.jpg'+'\n')
			count = count + 1


fp_log.close()
fp_w.close()