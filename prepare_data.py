import pandas as pd
import numpy as np

import os
import sys

import yaml
# 'data/iris.csv'
def load_data_with_preparing(path):
    iris = pd.read_csv(path)
    
    np.unique(iris['variety'])
    classes = {'Setosa' : 0, 'Versicolor' : 1, 'Virginica': 2}
    iris['variety'] = iris['variety'].map(classes)
    return iris

def main():
    
    dataset = load_data_with_preparing(sys.argv[1])

   

    os.makedirs(os.path.join('data','prepare'))

    pd.DataFrame(dataset).to_csv(os.path.join('data','prepare','data.csv')) 
if __name__ == "__main__":
    main()
