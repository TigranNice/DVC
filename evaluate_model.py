import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys 
import pickle
import json

import torch
import torch.nn as nn

class NeuroNet(nn.Module):
    def __init__(self, len_inp, len_out):        
        super(NeuroNet, self).__init__()     

        self.fc1 = nn.Linear(len_inp, 64)

        self.fc2 = nn.Linear(64, 32)

        self.fc3 = nn.Linear(32, len_out)
        
    def forward(self, x):                      
        x = self.fc1(x)                         
        x = nn.ReLU()(x)                   
        x = self.fc2(x)                          
        x = nn.ReLU()(x)      
        x = self.fc3(x)
        x = nn.ReLU()(x)   
        return x


def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)

def load_data(path):
    l_train = os.path.join(path, 'losses_train.csv')
    l_test = os.path.join(path, 'losses_test.csv')
    model = 0
    X_test = 0
    y_test = 0
    with open(os.path.join(path,'model.pkl'), "rb") as fl:
        model = pickle.load(fl)
    with open(os.path.join(path,'X_test.pkl'), "rb") as fl:
        X_test = pickle.load(fl)
    with open(os.path.join(path,'y_test.pkl'), "rb") as fl:
        y_test = pickle.load(fl)
    

    return model, pd.read_csv(l_train), pd.read_csv(l_test), X_test, y_test    

def accuracy_graph(model, X_test, y_test, path):
    predictions_test =  []
    with torch.no_grad():
        predictions_test = model(X_test)
    test_acc  = get_accuracy_multiclass(predictions_test,y_test)

    with open(os.path.join(path,'metrics.json'), 'w') as f:
        json.dump(f'{test_acc}', f)
    
def loss_graph(model, l_train, l_test, path):
    plt.figure(figsize=(10,10))
    
    l_train = np.delete(l_train, 0, axis = 1)
    l_test = np.delete(l_test, 0, axis = 1)
    
    plt.plot(l_train, label='train loss')
    plt.plot(l_test, label='test loss')
    plt.legend()
    plt.savefig(os.path.join(path,'losses.png'))


def main():
    os.makedirs(os.path.join('data','evaluate'))
    model, l_train, l_test, X_test, y_test = load_data(sys.argv[1])
    accuracy_graph(model, X_test, y_test, sys.argv[2])
    loss_graph(model, l_train, l_test, sys.argv[2])
if __name__ == '__main__':
    main()    