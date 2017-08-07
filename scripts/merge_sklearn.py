import pickle
import numpy as np
y_test_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_val_label_correct.txt"
score1 = pickle.load(open('./sklearn_model/vggfc6_prob_10_0.1', "r"))
score2 = pickle.load(open('./sklearn_model/vggfc7_prob_10_0.1', "r"))
score3 = pickle.load(open('./sklearn_model/res10_prob', "r"))
score4 = pickle.load(open('./sklearn_model/res10_prob_10_0.1', "r"))
score5 = pickle.load(open('./sklearn_model/lstm_01_prob', "r"))
score6 = pickle.load(open('./sklearn_model/lstm_01_82', "r"))
score7 = pickle.load(open('./sklearn_model/res34_prob_10_0.1', "r"))

score8 = pickle.load(open('./sklearn_model/c3d_afew_prob', "r"))
label = [line.strip() for line in  open(y_test_path,'r').readlines()]
count = 0
for i in range(len(label)):
	if np.argmax(2*score8[i]+score6[i]) == int(label[i]):
		count += 1
print "accuracy", float(count)/289

# if np.argmax(score1[i]+score2[i]+1.2*score5[i]+2*score3[i]+score4[i]) == int(label[i]):
#score1[i]+score3[i]+score2[i]+2*score4[i]+1.2*score5[i]+2*