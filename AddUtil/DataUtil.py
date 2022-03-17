import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

SEED = 42

class Process():
    
    @staticmethod
    def get_cat_var(data):

        cat_features_columns = []
        for item in data.columns:
            if data.dtypes[item]=='object':
                cat_features_columns.append(item)
        return cat_features_columns

    @staticmethod
    def get_missing_values(data):
        missing_values_columns = []
        missing_values = data.isnull().sum()
        for index, value in missing_values.items():
            if value != 0:
                missing_values_columns.append(index)
        return missing_values_columns

    @staticmethod
    def encode_cat_var(data, exclude_col=['time', 'expiration', 'age']):

        for column in Process.get_cat_var(data):
            if column not in exclude_col:
                data = pd.get_dummies(data, columns=[column])
    
        return data

    @staticmethod
    def fill_missing_val(data, eps=0.7):

        dataframe = data.copy()
        for column in Process.get_missing_values(dataframe):
            if dataframe[column].isna().sum() / dataframe.shape[0] > eps:
                dataframe.drop(labels=column, axis=1, inplace=True)
            elif column not in Process.get_cat_var(dataframe):
                dataframe[column].fillna(value=dataframe[column].mean(), inplace=True)
                
        return dataframe

    @staticmethod
    def get_label(data, label='Y'):
        
        y = np.array(data[label])
        return y
    
    
    @staticmethod
    def get_features(data, label=None):

        if label is not None:
            features = data.drop([label], axis=1)
        else: features = data
        return features

    @staticmethod
    def train_val_test_split(target, features, size=0.8, seed=SEED):

        X_train, X_rem, y_train, y_rem = train_test_split(features,target, train_size=size, random_state=seed)

        
        X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=seed)

        print(f"Size of the training samples   = {X_train.shape[0]}, {size*100}%")
        print(f"Size of the validation samples = {X_val.shape[0]}, {((1-size)/2)*100}%")
        print(f"Size of the test sample        = {X_val.shape[0]}, {((1-size)/2)*100}%")

        return (X_train, y_train, X_val, y_val, X_test, y_test)

    @staticmethod
    def determine_n_components(features, eps=0.999):
       
        pca = PCA().fit(features)
        exp_var = np.cumsum(pca.explained_variance_ratio_)

        number = 0
        for i in exp_var:
            number += 1
            if i >=eps:
                break

        return (number, exp_var)