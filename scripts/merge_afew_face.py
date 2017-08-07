import pickle
import numpy as np
fp = open('./lstm_afew_face/test_video_clip','r')
list = [line.strip() for line in fp.readlines()]
list.append('2575')
data = pickle.load(open('./sklearn_model/official_vgg_testfc6_prob','r'))
final_score=[]
for i in range(len(list)-1):
	count = int(list[i+1])-int(list[i])
	score=0
	for j in range(count):
		score = score + data[int(list[i])-1+j]
	score = score / count
	final_score.append(score)

final_data = np.array(final_score,dtype=float)

pickle.dump(final_data,open('lbp_test_prob','w'))

s = pickle.load(open('lbp_test_prob','r'))

print s[0]
print len(s)
print s[362]
print data[1393:1395]


