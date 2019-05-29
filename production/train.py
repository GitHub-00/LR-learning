import sklearn
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.externals import joblib
sys.path.append('../')
import util.get_feature_num as GF


def train_lr_model(train_file, model_coef, model_file, feature_num_file):
    '''
    Args:
        train_file: file for lr train
        model_coef: w1, w2,...
        model_file: model pkl
    '''
    #total_feature_num = 118
    total_feature_num = GF.get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=',', usecols=-1)
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=',', usecols=feature_list)
    lr_cf = LRCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5).fit(train_feature, train_label)
    scores = list(lr_cf.scores_.values())[0]
    print('diff: %s' %(",".join([str(ele) for ele in scores.mean(axis = 0)])))
    print('accuracy: %s (+-%0.2f)' %(scores.mean(), scores.std()*2))

    lr_cf = LRCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5, scoring='roc_auc').fit(train_feature, train_label)
    scores = list(lr_cf.scores_.values())[0]
    print('diff: %s' % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    print('auc: %s (+-%0.2f)' % (scores.mean(),scores.std()*2))

    coef = lr_cf.coef_[0]
    fw = open(model_coef,'w+')
    fw.write(','.join(str(ele) for ele in coef))
    fw.close()

    joblib.dump(lr_cf, model_file)


if __name__ == '__main__':
    train_lr_model('../data/train_file','../data/lr_coef','../data/lr_model_file','../data/feature_num')


