from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from training_functions import activation_function
import numpy as np

def sliding_window(
        x_data,
        y_data,
        window_size,
        stride=1
):
    ''' 
    perform a transformation in sliding window for time series

    Parameters:
    x_data - array - all of the predictors
    y_data - array - targets
    window_size - int - size of the window 
    stride - int - size of the target 

    Returns:
    two array; one for the predictors and one for the targets 
    '''
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    X, Y = [], []
    n = len(x_data)

    for i in range(0, n - window_size - stride + 1, stride):
        X.append(x_data[i : i + window_size])
        Y.append(y_data[i + window_size : i + window_size + stride])

    return np.array(X), np.array(Y)



def preprocess(
    df: pd.DataFrame,
    split_ratio_train_valid: float,
    split_ratio_train_test: float,
    device: str,
    window_size: int = 20,
    stride: int = 1,
    target_col: str = "OBS_VALUE",
    date_col: str = "TIME_PERIOD", 
    drop_cols: list = None):
    """
    Preprocessing data -> encoding for date, train/test spliting, train/valid spliting, scaling on train and sliding window

    Returns :
        X_train, X_val, X_test, y_train, y_valid, y_test
    """

    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day

    # cyclical encodings
    df["dayofyear_sin"] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365)

    df["month_sin"] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[date_col].dt.month / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df[date_col].dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df[date_col].dt.weekday / 7)

    # drop useless columns
    cols_to_drop = [date_col]
    if drop_cols:
        cols_to_drop += drop_cols

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # handle missing values
    df.fillna(0, inplace=True)

    # split X / y
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    nb_of_samples = len(X)

    X_train_full = X[:int(nb_of_samples * split_ratio_train_test):]
    X_test = X[int(nb_of_samples * split_ratio_train_test):]
    y_train_full = y[:int(nb_of_samples * split_ratio_train_test)]
    y_test = y[int(nb_of_samples * split_ratio_train_test):]

    nb_of_training_samples = len(X_train_full)

    X_train = X_train_full[:int(nb_of_training_samples * split_ratio_train_valid)]
    X_valid = X_train_full[int(nb_of_training_samples * split_ratio_train_valid):]
    y_train = y_train_full[:int(nb_of_training_samples * split_ratio_train_valid)]
    y_valid = y_train_full[int(nb_of_training_samples * split_ratio_train_valid):]

    # normalisation (fit on train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # sliding windows
    X_train, y_train = sliding_window(X_train, y_train, window_size, stride)
    X_valid,   y_valid  = sliding_window(X_valid,   y_valid,   window_size, stride)
    X_test,  y_test  = sliding_window(X_test,  y_test,  window_size, stride)

    X_train, X_valid = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_valid, dtype=torch.float32).to(device)
    X_test, y_train = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    y_valid, y_test = torch.tensor(y_valid, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


### ------------------------------------------------------- ###


class gru(nn.Module):
    def __init__(self, input_size, mode, hidden_size=64):
        ''' 
        Parameters:
        input_size  - int  - number of features
        hidden_size - int  - dimension of the hidden state 
        mode        - str  - activation function: 'relu' ou 'gelu'
        '''
        super().__init__()
        self.hidden_size = hidden_size

        if mode == "relu":
            self.activation = nn.ReLU()
        elif mode == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        Forward pass

        Parameters:
        x: tensor [batch, seq_len, input_size]

        Returns:
        out: tensor [batch, 1]
        '''
        out, _ = self.gru(x)          
        out = out[:, -1, :]           

        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)

        return out


### ------------------------------------------------------- ###


class lstm(nn.Module):
    def __init__(self, input_size, mode, hidden_size=64):
            ''' 
            Parameters:
            input_size - int - number of features
            hidden_size - int - dimension of the hidden state 
            activation - string - name of the activation function relu or gelu
            '''
            
            super().__init__()
            self.activation = activation_function(mode)
            self.mode = mode

            if mode == "relu":
                self.activation = nn.ReLU()
            elif mode == "gelu":
                self.activation = nn.GELU()
            else:
                raise ValueError("activation must be 'relu' or 'gelu'")
            
            # lstm layer
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

            # fully connected layers
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)  

    def forward(self, x):
        ''' 
        forward function

        Parameters:
        x: tensor [batch, seq_len, input_size]

        Returns:
        x: tensor[batch, 1, input_size]
        '''

        out, _ = self.lstm(x)
        out = out[:, -1, :]       

        out = self.fc1(out)
        out = self.activation(out)     
        out = self.fc2(out)            

        return out