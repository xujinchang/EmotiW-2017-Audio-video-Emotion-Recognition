
from lstm_architecture import one_hot, run_with_config

import numpy as np

import os

from read_emotion import load_Y_my, load_X_my

from jiangwei import do_pca, load_X_pca, reshape_pca

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        self.train_count = len(X_train)  # 451training series
        self.test_data_count = len(X_test)  # 36 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        self.n_classes = 7 # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 100
        self.batch_size = 1
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        # Dropout is added on inputs and after each stacked layers (but not
        # between residual layers).
        self.keep_prob_for_dropout = 0.85  # **(1/3.0)

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
        # self.n_layers_in_highway = 1  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        # self.n_stacked_layers = 1  # Stack multiple blocks of residual
        # layers.


#--------------------------------------------
# Dataset-specific constants and functions + loading
#--------------------------------------------

# Useful Constants

# Those are separate normalised input features for the neural network


# Load "X" (the neural network's training and testing inputs)
#X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfacetrain_fc6.fea"
#X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_our_face_correct/vgg16_afewfaceval_fc6.fea"
#y_train_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_train_label_correct.txt"
#y_test_path = "/home/xujinchang/share/caffe-center-loss/correct_lstm/lstm_val_label_correct.txt"

X_train_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_vgg16_face_train_fc6.fea"
X_test_signals_paths = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_vgg16_face_val_fc6.fea"
y_train_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_afewface_train_label_1.fea"
y_test_path = "/home/xujinchang/share/caffe-center-loss/lstm_afew_face/single_lbp_afewface_val_label_1.fea"
X_train_ori = load_X_pca(X_train_signals_paths)
X_test_ori = load_X_pca(X_test_signals_paths)
standard = StandardScaler()
#standard = Normalizer()
X_train_norm = standard.fit_transform(X_train_ori)
X_test_norm = standard.transform(X_test_ori)
pca = PCA(n_components=1024)

X_train=pca.fit_transform(X_train_norm)
X_test = pca.transform(X_test_norm)

X_train = reshape_pca(X_train)
X_test = reshape_pca(X_test)
y_train = one_hot(load_Y_my(y_train_path))
y_test = one_hot(load_Y_my(y_test_path))
#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

n_layers_in_highway = 0
n_stacked_layers = 1
trial_name = "{}x{}".format(n_layers_in_highway, n_stacked_layers)

for learning_rate in [0.0001]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for lambda_loss_amount in [0.005]:
        for clip_gradients in [15.0]:
            print "learning_rate: {}".format(learning_rate)
            print "lambda_loss_amount: {}".format(lambda_loss_amount)
            print ""

            class EditedConfig(Config):
                def __init__(self, X, Y):
                    super(EditedConfig, self).__init__(X, Y)

                    # Edit only some parameters:
                    self.learning_rate = learning_rate
                    self.lambda_loss_amount = lambda_loss_amount
                    self.clip_gradients = None
                    # Architecture params:
                    self.n_layers_in_highway = n_layers_in_highway
                    self.n_stacked_layers = n_stacked_layers

            # # Useful catch upon looping (e.g.: not enough memory)
            # try:
            #     accuracy_out, best_accuracy = run_with_config(EditedConfig)
            # except:
            #     accuracy_out, best_accuracy = -1, -1
            accuracy_out, best_accuracy, f1_score_out, best_f1_score = (
                run_with_config(EditedConfig, X_train, y_train, X_test, y_test, learning_rate)
            )
            print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)

            with open('{}_result_emotion_7.txt'.format(trial_name), 'a') as f:
                f.write(str(learning_rate) + ' \t' + str(lambda_loss_amount) + ' \t' + str(clip_gradients) + ' \t' + str(
                    accuracy_out) + ' \t' + str(best_accuracy) + ' \t' + str(f1_score_out) + ' \t' + str(best_f1_score) + '\n\n')

            print "________________________________________________________"
        print ""
print "Done."
