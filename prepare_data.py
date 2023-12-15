import pandas as pd
import numpy as np

import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import yaml
# 'data/iris.csv'
def load_data_with_preparing(path):
    iris = pd.read_csv(path)

    np.unique(iris['variety'])
    classes = {'Setosa' : 0, 'Versicolor' : 1, 'Virginica': 2}
    iris['variety'] = iris['variety'].map(classes)
    return iris

def splitting(dataset, split_percantage):
    X = dataset.drop('variety', axis=1)
    y = dataset['variety']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = split_percantage, random_state=0)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    
    dataset = load_data_with_preparing(sys.argv[1])

    X_train, X_test = splitting(dataset, params[test_size])

    os.makedirs(os.path.join('data','prepare'))

    X_train.to_csv(os.path.join('data','prepare','train.csv')) 
    X_test.to_csv(os.path.join('data','prepare','test.csv')) 
if __name__ == "__main__":
    main()
