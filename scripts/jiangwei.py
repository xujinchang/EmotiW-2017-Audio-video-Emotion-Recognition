from sklearn.decomposition import PCA
import numpy as np


def load_X_pca(X_signals_paths):
    X_signals = []
    file = open(X_signals_paths, 'r')
    for row in file:
    	X_signals.append([np.array(serie, dtype=np.float32) for serie in row.strip().split(' ')])

    file.close()
    return np.array(X_signals)

def do_pca(X_data):
	pca=PCA(n_components=1024)
	newX=pca.fit_transform(X_data)
	new_feature = reshape_pca(newX)
	return new_feature


def reshape_pca(newX):
	count = 0
	item = []
	feature = []
	for index in xrange(0,newX.shape[0]):
		count = count + 1
		item.append(newX[index])

		if count % 128 == 0:
			feature.append(item)
			item = []
	return np.array(feature)





# X_train_signals_paths = "/home/xujinchang/caffe-blur-pose/valid_fc7_feature_new.fea"

# X_data = load_X_my(X_train_signals_paths)
# new_feature = do_pca(X_data)

# print new_feature.shape
