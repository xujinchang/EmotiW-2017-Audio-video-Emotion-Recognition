
from lstm_architecture_test import one_hot, run_with_config, test_with_config

import numpy as np

import os

from read_emotion import load_Y_my, load_X_my


from jiangwei import do_pca, load_X_pca, reshape_pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#--------------------------------------------
# Neural net's config.
#--------------------------------------------

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Data shaping
        self.train_count = len(X_train)  # 451 training series
        self.test_data_count = len(X_test)  # 36 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        self.n_classes = 7  # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 10000
        self.batch_size = 1
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        # Dropout is added on inputs and after each stacked layers (but not
        # between residual layers).
        self.keep_prob_for_dropout = 0.5  # **(1/3.0)0.85

        # Linear+relu structure
        self.bias_mean = 0.3
        # I would recommend between 0.1 and 1.0 or to change and use a xavier
        # initializer
        self.weights_stddev = 0.2

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        # Features count is of 9: three 3D sensors features over time
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 128  # nb of neurons inside the neural network
        # Use bidir in every LSTM cell, or not:
        self.use_bidirectionnal_cells = False

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        self.n_layers_in_highway = 3  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        self.n_stacked_layers = 3  # Stack multiple blocks of residual
        # layers.


#--------------------------------------------
# Dataset-specific constants and functions + loading
#--------------------------------------------

# Useful Constants

# Those are separate normalised input features for the neural network

#X_train_signals_paths = "/home/xujinchang/caffe-blur-pose/valid_fc7_feature_new.fea"
#X_test_signals_paths = "/home/xujinchang/caffe-blur-pose/test_fc7_feature_new.fea"
#y_train_path = "/home/xujinchang/caffe-blur-pose/valid_y_label_2.fea"
#y_test_path = "/home/xujinchang/caffe-blur-pose/valid_y_label_2.fea"
#X_train = load_X_my(X_train_signals_paths)
#X_train_signals_paths = "/localSSD/xjc/codalab_train/train/train_train_fc7_feature_new.fea"
#X_test_signals_paths = "/localSSD/xjc/codalab_train/train/train_valid_fc7_feature_new.fea"
#y_train_path = "/localSSD/xjc/codalab_train/train/train_y_label_2.fea"
#y_test_path = "/localSSD/xjc/codalab_train/train/valid_y_label_2.fea"

#X_test_signals_paths = "/localSSD/xjc/codalab_train/valid/valid_fc7_feature_new.fea"
#X_valid_path = "/localSSD/xjc/codalab_train/test/final_fc7_feature_new.fea"
#X_test = load_X_my(X_test_signals_paths)

X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfacetrain_fc6.fea"
#X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfaceval_fc6.fea"
#y_train_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_train_label_correct.txt"
#y_test_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_val_label_correct.txt"
#X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_vgg16_face_train_fc6.fea"
#X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_vgg16_face_val_fc6.fea"
# X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_resnet/res34_afewfacetrain.fea"
# X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_resnet/res34_afewfaceval.fea"
y_train_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_train_label.fea"
y_test_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/lbp_afewface_val_label.fea"
#y_test_path ="/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_test_fake_label"
X_test_signals_paths = '/home/xujinchang/share/caffe-center-loss/lstm_afew_face/our_vgg16_face_test_fc6.fea'

X_train_ori = load_X_pca(X_train_signals_paths)
X_test_ori = load_X_pca(X_test_signals_paths)
standard = StandardScaler()
X_train_norm = standard.fit_transform(X_train_ori)
X_test_norm = standard.transform(X_test_ori)
pca = PCA(n_components=1024)
X_train=pca.fit_transform(X_train_norm)
X_test = pca.transform(X_test_norm)

#X_train = reshape_pca(X_train_norm)
X_valid_result = reshape_pca(X_test)
#y_train = one_hot(load_Y_my(y_train_path))
y_test = one_hot(load_Y_my(y_test_path))

#X_valid_result = load_X_pca(X_valid_path)
#X_valid_result = do_pca(X_valid_result)
#X_valid_result = load_X_my(X_valid_path)
#X_test = do_pca(X_test)
    # y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    # y_test_path = DATASET_PATH + TEST + "y_test.txt"
#y_train = one_hot(load_Y_my(y_train_path))
#y_test = one_hot(load_Y_my(y_test_path))
#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

n_layers_in_highway = 0
n_stacked_layers = 1
trial_name = "{}x{}".format(n_layers_in_highway, n_stacked_layers)
class EditedConfig(Config):
    def __init__(self, X, Y):
        super(EditedConfig, self).__init__(X, Y)
        self.n_layers_in_highway = n_layers_in_highway
        self.n_stacked_layers = n_stacked_layers

pred_out = test_with_config(EditedConfig, X_valid_result, y_test)
print type(pred_out)
print pred_out[0].shape
print pred_out[0]

fx = open('our_final_test','w')
fy = open('test_our_label_result','w')
import pickle
pickle.dump(pred_out,fx)

for item in xrange(0,len(pred_out)):
    fy.write(str(np.argmax(pred_out[item]))+'\n')

fx.close()
fy.close()

# for learning_rate in [0.001]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
#     for lambda_loss_amount in [0.005]:
#         for clip_gradients in [15.0]:
#             print "learning_rate: {}".format(learning_rate)
#             print "lambda_loss_amount: {}".format(lambda_loss_amount)
#             print ""

#             class EditedConfig(Config):
#                 def __init__(self, X, Y):
#                     super(EditedConfig, self).__init__(X, Y)

#                     # Edit only some parameters:
#                     self.learning_rate = learning_rate
#                     self.lambda_loss_amount = lambda_loss_amount
#                     self.clip_gradients = clip_gradients
#                     # Architecture params:
#                     self.n_layers_in_highway = n_layers_in_highway
#                     self.n_stacked_layers = n_stacked_layers

#             # # Useful catch upon looping (e.g.: not enough memory)
#             # try:
#             #     accuracy_out, best_accuracy = run_with_config(EditedConfig)
#             # except:
#             #     accuracy_out, best_accuracy = -1, -1
#             accuracy_out, best_accuracy, f1_score_out, best_f1_score = (
#                 run_with_config(EditedConfig, X_train, y_train, X_test, y_test)
#             )
#             print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)

#             with open('{}_result_emotion_12.txt'.format(trial_name), 'a') as f:
#                 f.write(str(learning_rate) + ' \t' + str(lambda_loss_amount) + ' \t' + str(clip_gradients) + ' \t' + str(
#                     accuracy_out) + ' \t' + str(best_accuracy) + ' \t' + str(f1_score_out) + ' \t' + str(best_f1_score) + '\n\n')

#             print "________________________________________________________"
#         print ""
# print "Done."
