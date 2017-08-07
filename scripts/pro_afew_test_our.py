import os
path = '/home/tmp_data_dir/emotj2017/AFEW/Test/'
path1 = '/home/tmp_data_dir/emotj2017/AFEW/Test_crop'

video_name = os.listdir(path)

for item in video_name:
	os.system("cd {name}".format(name = path))
	os.system("mkdir {name1}".format(name1 = item))