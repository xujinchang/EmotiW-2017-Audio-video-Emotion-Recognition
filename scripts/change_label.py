import os
# fp = open('lstm_val.txt','r')
fp1 = open('lstm_train_correct.txt','r')
fp2 = open('lstm_train_label_correct.txt','w')
label_list = ['Sad','Surprise','Fear','Angry','Disgust','Neutral','Happy']


count = 0
for line in fp1.readlines():
	image_name = line.strip().split(' ')
	item = image_name[1]
	count += 1
	if count % 16 == 0:	
		fp2.write(item+'\n')


fp1.close()
fp2.close()
