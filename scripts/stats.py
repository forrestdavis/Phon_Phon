from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

def load_data(data_dir='../data_ternary/'):

    train_data = open(data_dir+'train_pad.data', 'r')

    #Get x and y train values
    temp_x = []
    temp_y = []
    for line in train_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for all sounds
        x_val = line[2:len(line)-1]

        #Join all x values
        x_val = ' '.join(x_val)
        x_val = x_val.split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]
        temp_x.append(x_val)

        if int(line[len(line)-1]) == 0:
            temp_y.append(1)
        else:
            temp_y.append(0)

        #temp_y.append(int(line[len(line)-1]))

    train_data_x = np.array(temp_x)
    train_data_y = np.array(temp_y)

    return train_data_x, train_data_y

def load_data_last_sound(data_dir='../data_ternary/'):

    #train_data = open(data_dir+'class_1/train_f.data', 'r')
    train_data = open(data_dir+'train.data', 'r')

    #Get x and y train values
    temp_x = []
    temp_y = []
    for line in train_data:
        line=line.strip()
        if not line:
            continue
        line = line.split('\t\t')

        #Get binary feature values for last sound
        x_val = line[len(line)-2:len(line)-1]
        x_val = x_val[0].split()
        
        #Turn list of strings into list of ints
        x_val = [int(x) for x in x_val]

        temp_x.append(x_val)

        if int(line[len(line)-1]) == 1:
            temp_y.append(1)
        else:
            temp_y.append(0)

    train_data_x = np.array(temp_x)
    train_data_y = np.array(temp_y)

    return train_data_x, train_data_y

def get_importances():

    #X, y = load_data()
    X, y = load_data_last_sound()

    '''
    X_reduced = []
    for x in X:
        X_reduced.append(x[320:])
    X_reduced = np.array(X_reduced)
    '''
    
    X_reduced = X

    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

    forest.fit(X_reduced, y)
    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_], 
            axis=0)

    indices = np.argsort(importances)[::-1]

    features = ['High', 'Back', 'Low', 'ATR', 'Round', 
            'Syllabic', 'Cons',
            'Son', 'Cont', 'Nasal', 'Lateral', 'DR', 'Voice', 
            'Labial', 'Coronal', 'Dorsal', 'Laryngeal', 'Anterior',
            'Dist', 'Strident']

    sub_features = []
    sound_number = []
    for stuff in indices:
        sub_features.append(features[stuff % 20])
        if indices.shape[0] == 20:
            sound_number.append(17)
        else:
            sound_number.append(stuff / 20 + 1)

    print("Feature ranking:")

    for f in range(X_reduced.shape[1]):
        print('%d. feature %s on sound %d (%f)'%(f+1, sub_features[f], 
            sound_number[f], importances[indices[f]]))

    plt.figure()
    plt.title("Feature Importances")

    plt.bar(range(X_reduced.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_reduced.shape[1]), sub_features)
    plt.xlim([-1, X_reduced.shape[1]])
    plt.show() 

if __name__ == '__main__':
    get_importances()
