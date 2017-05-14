# coding = utf-8

import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pylab as plt
import re

def load_data(filename):
    data = []
    target = []
    with open(filename, "r") as fr:
        fr.readline()
        for line in fr:
            line = re.split(",", line)
            date_time = ",".join(line[0:2])
            gate_dir = line[3]
            label = line[-1].strip('\n')
            data.append(line[:-1])
            target.append(label)
    return data, target

def MAPE(save):
    mape = 0
    for key in save:
        count = 0
        for item in save[key]:
            count += abs(item[0] - item[1])
        mape += count*1.0 / len(save[key])
    mape /= 1.0 * len(save)
    return mape

if __name__ == "__main__":
    filename = 'features_train(with head).csv'
    # filename = 'test.csv'
    train_data, train_target = load_data(filename)
    X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(train_data, train_target,
                                                                           test_size=0.15, random_state=0)

    train_id = [line[:3] for line in X_train]
    valid_id = [line[:3] for line in X_valid]
    X_train = [line[3:] for line in X_train]
    X_valid = [line[3:] for line in X_valid]

    gbdt = GradientBoostingClassifier()
    gbdt.fit(X_train, Y_train)
    GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
                               max_depth=6, max_features=None, max_leaf_nodes=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               random_state=None, subsample=1.0, verbose=0,
                               warm_start=False)
    gbdt_predict_labels = gbdt.predict(X_valid)

    presave = {}
    with open("valid_res.csv", 'w') as fw:
        for index in range(len(X_valid)):
                fw.write("{}, {}, {}\n".format(valid_id[index], Y_valid[index], gbdt_predict_labels[index]))
                if valid_id[index][2] not in presave:
                    presave[valid_id[index][2]] = []
                    presave[valid_id[index][2]].append([float(Y_valid[index]), float(gbdt_predict_labels[index])])
                else:
                    presave[valid_id[index][2]].append([float(Y_valid[index]), float(gbdt_predict_labels[index])])

    print(MAPE(presave))

