import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import sys
import yaml
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

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


def load_data(path):
    data_input = os.path.join(path, 'data.csv')

    return pd.read_csv(data_input) 

def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses):
    
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)
        


        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        
        output_test = model(X_test)
        loss_test = criterion(output_test,y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()


def train_model(dataset, epochs, learning_rate, _test_size):       
    X = dataset.drop('variety', axis=1)
    X = np.delete(X,0 ,axis = 1)
    y = dataset['variety']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_test_size, random_state=0)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)

    model = NeuroNet(4,3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    train_losses = np.zeros(epochs)
    test_losses  = np.zeros(epochs)

    train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,epochs,train_losses,test_losses)

    return model, train_losses, test_losses, X_test, y_test
def main():
    params = yaml.safe_load(open("params.yaml"))['train']

    os.makedirs(os.path.join('data','train'))

    dataset = load_data(sys.argv[1])

    model, train_losses, test_losses, X_test, y_test = train_model(dataset, params['num_epochs'], params['learning_rate'], params['test_size'])

    output = os.path.join(sys.argv[2], 'model.pkl')
    output_losses_train = os.path.join(sys.argv[2], 'losses_train.csv')
    output_losses_test = os.path.join(sys.argv[2], 'losses_test.csv')
    output_X_test = os.path.join(sys.argv[2], 'X_test.pkl')
    output_y_test = os.path.join(sys.argv[2], 'y_test.pkl')
    with open(output, "wb") as fd:
        pickle.dump(model, fd)
    with open(output_X_test, "wb") as fd:
        pickle.dump(X_test, fd)
    with open(output_y_test, "wb") as fd:
        pickle.dump(y_test, fd)    
    pd.DataFrame(train_losses).to_csv(output_losses_train)
    pd.DataFrame(test_losses).to_csv(output_losses_test)


if __name__ == '__main__':
    main()        