import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_decomposition import CCA
from sklearn.lda import LDA
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals import joblib
from sklearn import grid_search
import time
def load_x_data(filename):
    feature = []
    file = open(filename, 'r')
    for row in file:
    	feature.append([np.array(serie, dtype=np.float32) for serie in row.strip().split(' ')])
    file.close()
    return np.array(feature)


def load_y_data(filename):
	label = []
	file = open(filename,'r')
	for row in file:
		label.append([np.array(serie, dtype=np.float32) for serie in row.strip().split(' ')])
	file.close()
	return np.array(label)

def reshape_pca(newX):
	count = 0
	item = []
	feature = []
	for index in xrange(0,newX.shape[0]):
		count = count + 1
		item.append(newX[index])

		if count % 16 == 0:
			feature.append(item)
			item = []
	return np.array(feature)

##wu jiandu
def do_PCA(X_data):
	pca=PCA(n_components=1024) #512,128
	newX=pca.fit_transform(X_data)
	return newX
# def do_LDA():


def do_l2norm(X_data):
	x_normalized=preprocessing.normalize(X_data,norm='l2')
	return x_normalized

#svm = SGDClassifier(loss = 'hinge')
#https://ljalphabeta.gitbooks.io/python-/content/kernelsvm.html

def use_SVM(X_data,y_data):
	p_gamma = 0.1
	p_C = 10
	svm = SVC(kernel = 'rbf',random_state=0, gamma=p_gamma ,C=p_C, probability=True)
	svm.fit(X_data,y_data)
	joblib.dump(svm,"./sklearn_model/svm_trainval1_{param1}_{param2}".format(param1 = p_gamma,param2 = p_C))
	return svm

def use_tree(X_data,y_data):
	tree = DecisionTreeClassifier(criterion='entropy',max_features='sqrt',max_depth=3,random_state=0)
	tree.fit(X_data,y_data)
	return tree
	# hua tu
	# X_comined = np.vstack((X_data,X_valid))
	# y_comined = np.hstack((y_data,y_valid))
	# plot_decision_regions(X_comined,y_comined,classifier=tree,test_idx=range(105,150))
	# plt.show()
	# export_graphviz(tree,out_file='tree.dot',feature_names = ['petal length','petal width'])

def use_SGD(X_data,y_data):
	clf = SGDClassifier(loss="hinge", penalty="l2")
	clf.fit(X_data, y_data)
	return clf

# def use_KNN(X_data,y_data):





# def use_RandomForest(X_data,y_data):



def load_X(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()


    return np.concatenate((X_signals[0],X_signals[1],X_signals[2],X_signals[3],X_signals[4],X_signals[5],X_signals[6]),axis=0)
def load_Y(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.int) for serie in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()


    return np.concatenate((X_signals[0],X_signals[1],X_signals[2],X_signals[3],X_signals[4],X_signals[5],X_signals[6]),axis=0)
if __name__ == '__main__':
# 	TRAIN_TYPES = [
# 	"train_Sad0.fea",
# 	"train_Surprise1.fea",
# 	"train_Fear2.fea",
# 	"train_Ang3.fea",
#     "train_Disgust4.fea",
#     "train_Neu5.fea",
#     "train_Happy6.fea"
# 	]

# # Output classes to learn how to classify
# 	VAL_TYPES = [
# 	"val_Sad0.fea",
# 	"val_Surprise1.fea",
# 	"val_Fear2.fea",
# 	"val_Ang3.fea",
# 	"val_Disgust4.fea",
# 	"val_Neu5.fea",
# 	"val_Happy6.fea"
# 	]

# 	DATASET_PATH = "/home/xujinchang/share/caffe-center-loss/vgg16_feature_fc6/"
# 	TRAIN = "vgg16_fc6_"
# 	TEST = "vgg16_fc6_"
# 	TRAIN_LABEL = "train_fea/"
# 	VAL_LABEL = "val_fea/"
# 	start = time.time()
# 	X_train_fea = [DATASET_PATH + TRAIN + signal for signal in TRAIN_TYPES]
# 	X_val_fea = [DATASET_PATH + TEST + signal for signal in VAL_TYPES]

# 	X_train = load_X(X_train_fea)
# 	X_val = load_X(X_val_fea)

# 	y_train_fea = [DATASET_PATH + TRAIN_LABEL + signal for signal in TRAIN_TYPES]
# 	y_val_fea = [DATASET_PATH + VAL_LABEL + signal for signal in VAL_TYPES]

# 	Y_train = load_Y(y_train_fea)
# 	Y_val = load_Y(y_val_fea)
	start = time.time()
	##############################################our_face_train##############################################
	#X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfacetrain_fc6.fea"
	##X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfaceval_fc6.fea"
	#y_train_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_train_label_correct.txt"
	#y_test_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_val_label_correct.txt"
	X_submit_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_vgg16_face_test_fc6.fea"

#############################################official_face_train#############################################
	#X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_vgg16_face_train_fc6.fea"
	#X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_vgg16_face_val_fc6.fea"
	#y_train_path = '/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_train_label_1.fea'
	#y_test_path = '/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_val_label_1.fea'
	#X_submit_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/official_vgg16_face_test_fc6.fea"
	# X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_resnet/res34_afewfacetrain.fea"
	# X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_resnet/res34_afewfaceval.fea"

	# X_train_signals_paths = "/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/lbp_train.fea"
	# X_test_signals_paths = "/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/lbp_val.fea"
	# y_train_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_afewface_train_label_1.fea"#danyi label
	# y_test_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_afewface_val_label_1.fea"

	# X_train_signals_paths = "/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/c3d_train_fc7.fea"
	# X_test_signals_paths = "/home/xujinchang/share/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/script/c3d_val_fc7.fea"

	# X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/res10_lbp_face_train.fea"
	# X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/res10_lbp_face_val.fea"
	# y_train_path = '/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_train_label_1.fea'
	# y_test_path = '/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_val_label_1.fea'

	##############################our_train+val######################################################
	X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_trainval_vgg16_fc6.fea"
	X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_val2_vgg16_fc6.fea"
	y_train_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_trainval_label.fea"
	y_test_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_val2_label.fea"

	X_train_ori = load_x_data(X_train_signals_paths)
	X_test_ori = load_x_data(X_test_signals_paths)
	X_submit_ori = load_x_data(X_submit_path)

	Y_train = load_x_data(y_train_path)
	Y_test = load_x_data(y_test_path)
	#standard = StandardScaler()
	print "standard..."
	standard = Normalizer()
	X_train_norm = standard.fit_transform(X_train_ori)
	X_test_norm = standard.transform(X_test_ori)
	X_submit_norm = standard.transform(X_submit_ori)

	print "PCA..."
	pca = PCA(n_components=1024)
	X_train=pca.fit_transform(X_train_norm)
	X_test = pca.transform(X_test_norm)
	X_submit = pca.transform(X_submit_norm)


	print "reshape..."
	X_train_shape = reshape_pca(X_train)
	X_train = np.reshape(X_train_shape,(838,-1)) #744#3045#838#577
	X_test_shape = reshape_pca(X_test)
	X_test = np.reshape(X_test_shape,(28,-1))  #363#1408#28#289
	X_submit_shape = reshape_pca(X_submit)
	X_submit = np.reshape(X_submit_shape,(616,-1)) #
	# pca1 = PCA(n_components=1024)
	# X_train=pca.fit_transform(X_train)
	# X_test = pca.transform(X_test)

	# X_train = X_train_norm
	# X_test = X_test_norm

	end = time.time()
	print X_train.shape
	print Y_train.shape
	print "load time", end-start
	# standard = StandardScaler()
	# X_train_norm = standard.fit_transform(X_train)
	# X_val_norm = standard.transform(X_val)
	# normalize = preprocessing.Normalizer()
	# X_train_norm = normalize.fit_transform(X_train)
	# X_val_norm = normalize.transform(X_val)

	# end1 = time.time()
	# print "preprocessing time",end1-end


	print "use_SVM..."
	clf = use_SVM(X_train,Y_train.ravel())
	#clf = use_tree(X_train,Y_train.ravel())
	#classif = OneVsRestClassifier(SVC(kernel = 'rbf',random_state=0, gamma=1.0,C=10.0))
	#clf.fit(X_train,Y_train.ravel())
	end2 = time.time()
	print "train time",end2-end
	# score1 = classif.score(X_train_norm,X_val)
	# score2 = classif.score(X_val_norm,Y_val)
	#y_pred = clf.predict(X_test)
	#print(classification_report(y_true=Y_test, y_pred=y_pred))
	print "predict..."
	score = clf.score(X_test,Y_test)
	prob = clf.predict_proba(X_test)
	print prob
	prob_test = clf.predict_proba(X_submit)
	pickle.dump(prob, open('./sklearn_model'+'/'+'official_train_prob', "w"))
	pickle.dump(prob_test, open('./sklearn_model'+'/'+'our_vgg_testfc6_prob', "w"))
	end3 = time.time()
	print "train time",end3-end2
	print score
	print clf
    # classif.fit(X_train_norm, Y_train)
    # classif.score(X_val_norm,y_val)
