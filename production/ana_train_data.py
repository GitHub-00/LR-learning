
import sys
import os
import numpy as np
import pandas as pd


def get_input(input_train_file, input_test_file):
    '''
    Return:
        pd.Dataframe train_file
        pd.Dataframe test_file
    '''
    dtype_dict = {'age':np.int32,
                  'education-num':np.int32,
                  'captial-gain':np.int32,
                  'hours-per-week':np.int32,
                  'captial-gain':np.int32}
    use_list = list(range(15))
    use_list.remove(2)
    train_data_df = pd.read_csv(input_train_file, sep=",", header=0, dtype=dtype_dict, na_values='Nan', usecols=use_list)
    print(train_data_df.shape)
    train_data_df = train_data_df.dropna(axis=0, how='any')
    print(train_data_df.shape)
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how='any')
    return train_data_df, test_data_df

def ana_train_data(input_train_dat,
                       input_test_data,
                       output_train_file,
                       output_test_file):
    '''
    '''
    get_input(input_train_dat, input_test_data)

if __name__=='__main__':
    print(pd.__version__)
    ana_train_data('../data/train.txt','../data/test.txt','','')
