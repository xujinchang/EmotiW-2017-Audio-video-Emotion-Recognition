import numpy as np
import os

def load_X_my(X_signals_paths):
    X_signals = []
    item = []
    count = 0
    file = open(X_signals_paths, 'r')
    for row in file:
    	count = count + 1
    	item.append([np.array(serie, dtype=np.float32) for serie in row.strip().split(' ')])
    	if count % 128 == 0:
    		X_signals.append(item)
    		item=[]
    file.close()
    return np.array(X_signals)

def load_Y_my(X_signals_paths):
    X_signals = []
    file = open(X_signals_paths, 'r')
    for row in file:

    	X_signals.append([np.array(serie, dtype=np.float32) for serie in row.strip().split(' ')])

    file.close()
    return np.array(X_signals)
def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


    # X_signals.append(
    #         [np.array(serie, dtype=np.float32) for serie in [
    #             row.replace('  ', ' ').strip().split(' ') for row in file
    #         ]]
    #     )
    # file.close()
    # file = open(X_signals_paths, 'r')

    # X_signals.append(
    #         [np.array(serie, dtype=np.float32) for serie in [
    #             row.replace('  ', ' ').strip().split(' ') for row in file
    #         ]]
    #     )

    # file.close()
    print (np.array(X_signals).shape)
    return np.array(X_signals)    #(n,128,4096)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

