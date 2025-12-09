from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from training_functions import activation_function


def brevage_preprocessing(df: pd.DataFrame, val_size, test_size,random_state=1) -> tuple:
    df = df.copy()
    # date processing
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['year'] = df['Order_Date'].dt.year
    df['month'] = df['Order_Date'].dt.month
    df['day'] = df['Order_Date'].dt.day

    
    # one hot encoding for Region, Category, Product,Customer_Type
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols = ['Region', 'Category', 'Product', 'Customer_Type']
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=categorical_cols, inplace=True)
    
    # categorical to numerical for Customer_ID
    df['Customer_ID'] = df['Customer_ID'].astype('category').cat.codes
    
    # Dropping useless columns
    df.drop(columns=['Order_Date','Order_ID'], inplace=True)
    # Handle missing values by filling with 0
    df.fillna(0, inplace=True)
    
 

    X_brevage = df.drop('Total_Price', axis=1)
    y_brevage = df['Total_Price']
    
    assert val_size + test_size < 1

    val_size_adjusted = val_size / (1 - test_size)
    X_temp, X_test, y_temp, y_test = train_test_split(X_brevage, y_brevage, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = BrevageDataset(X_train_scaled, y_train)
    val_dataset = BrevageDataset(X_val_scaled, y_val)
    test_dataset = BrevageDataset(X_test_scaled, y_test)

    return train_dataset, val_dataset, test_dataset


class BrevageDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    def count_features(self):
        return self.features.shape[1]
    


class Brevage_model(nn.Module):
    def __init__(self, input_dim, mode,use_batch_norm=False):
        super(Brevage_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64) if use_batch_norm else None
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32) if use_batch_norm else None
        self.fc3 = nn.Linear(32, 1)
        self.activation = activation_function(mode)
        
        self.mode = mode
        self.use_batch_norm = use_batch_norm
    def forward(self, x):
        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
    

