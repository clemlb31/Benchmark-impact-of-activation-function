from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
import torch.nn.functional as F
from training_functions import activation_function
from torch.utils.data import TensorDataset

def mnist_preprocessing(X_train_full, y_train_full, X_test, y_test, val_size, random_state=1) -> tuple:
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, random_state=random_state)
    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 28 * 28)).reshape(-1, 28, 28)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 28 * 28)).reshape(-1, 28, 28)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 28 * 28)).reshape(-1, 28, 28)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long).view(-1))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.long).view(-1))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.long).view(-1))
    

    return train_dataset, val_dataset, test_dataset, scaler

class mnist_model(nn.Module):
    def __init__(self, mode, dropout_rate=0.5, use_batch_norm=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else self.identity
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else self.identity
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else self.identity
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batch_norm else self.identity
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)
        
        self.mode = mode
        self.use_batch_norm = use_batch_norm
        self.activation = activation_function(mode)

    def identity(self, x):
        return x
    def forward(self, x):

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        

        x = x.view(x.size(0), -1)
        

        x = self.activation(self.fc1(x))
        x = self.dropout3(x)
        x = self.activation(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)

        return x