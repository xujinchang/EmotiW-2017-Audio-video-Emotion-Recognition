import pickle
import numpy as np
fp = open('/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_val2_label.fea','r')
fp1 = open('emoti_final_result_3','w')
#fp = open('','r')
data = pickle.load(open('lstm_final_score_txt','r'))#lstm39.994

score1= pickle.load(open('./sklearn_model/single_face_vggfc6_prob_15_0.1', "r"))#single_fc637.19
#score2 = pickle.load(open('final_score_txt', "r"))#all_fc6#39.94
score2 = pickle.load(open('./sklearn_model/lbp_test_prob', "r"))#all_fc6#39.94
#score3 = pickle.load(open('./sklearn_model/c3d_afew_lbp_val.txt','r'))#c3d#41.32
score3 = pickle.load(open('./sklearn_model/c3d_lbp_test_171_result.txt','r'))#c3d#41.32
#score4 = pickle.load(open('map_to_afew_vggfc6_prob_10_0.1.txt','r'))#our#31.1
score4 = pickle.load(open('map_to_ourtrainval_testfc6','r'))
#score4 = pickle.load(open('./sklearn_model/map_to_lbp_vgg','r'))#our#31.1
#score5 = pickle.load(open('map_to_afew_lstm_01_82','r'))#our#32.5afew_checkpoint_01_fc61024_sub0.0001/82
score5 = pickle.load(open('./sklearn_model/map_to_lbp_lstm','r'))
score6 = pickle.load(open('./sklearn_model/lbp_prob','r'))#lbp36.6
score7 = pickle.load(open('all_fc7_final_score_txt','r'))#all_fc7 37.4
score8 = pickle.load(open('c3dfc7_prob_4096','r')) #38.56
score9 = pickle.load(open('lstm_145_prob','r'))#lstm_145 #38.8
score10 = pickle.load(open('res10_prob','r'))#33.05
#score11 = pickle.load(open('./sklearn_model/c3d_afew_lbp_128_val.txt','r'))#c3d128 39.66/good_model-1200
score11 = pickle.load(open('./sklearn_model/c3d_lbp_test_128_result.txt','r'))#c3d128 39.66/good_model-1200
score12 = pickle.load(open('./sklearn_model/c3d_afew_val36_171_result.txt','r'))#c3d128 38.01./model/lr_fixed/afew_conv3d_sport1m_0_0_128_171_0001_iter_2000
score13 = pickle.load(open('./sklearn_model/trainval1_prob','r'))
list = [line.strip() for line in fp.readlines()]
count = 0
for i in range(653):
	# if np.argmax(4*score2[i]+3.6*score3[i]+1.6*score4[i]+1.5*score5[i]+2.5*score11[i]) == int(list[i]):
	# 	count += 1
	fp1.write(str(np.argmax(3*score2[i]+3.5*score3[i]+3*score4[i]+1.5*score5[i]+2.5*score11[i]))+'\n')



#print float(count)/28
#363
fp.close()
fp1.close()
#3*score2[i]+3.5*score3[i]+score4[i]+score5[i] 0.468319559229
#(3*score2[i]+3.5*score3[i]+score4[i]+score5[i]+2*score11[i] 0.473829201102
#3*score2[i]+3.5*score3[i]+1.5*score4[i]+1.5*score5[i]+2.5*score11[i] 47.65 res2
#4*score2[i]+3.6*score3[i]+1.6*score4[i]+1.5*score5[i]+2.5*score11[i]
