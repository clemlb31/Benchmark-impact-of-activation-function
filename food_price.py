from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from training_functions import activation_function
import numpy as np
from torch.utils.data import TensorDataset
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
    device = torch.device("cpu"),
    window_size: int = 20,
    stride: int = 1,
    target_col: str = "OBS_VALUE",
    date_col: str = "TIME_PERIOD", 
    drop_cols: list = None,
    add_lag_features: bool = True,
    add_rolling_features: bool = True,
    normalize_target: bool = False):
    """
    Preprocessing amélioré avec features supplémentaires et meilleure gestion des données
    
    Parameters:
        add_lag_features: Ajouter des features lag (valeurs précédentes de la target)
        add_rolling_features: Ajouter des rolling statistics (moyennes mobiles)
        normalize_target: Normaliser la target (utile pour réduire l'échelle)
    
    Returns :
        X_train, X_val, X_test, y_train, y_valid, y_test, scaler_X, scaler_y (si normalize_target=True)
    """
    

    df = df.copy()
    df = df[df['REF_AREA_LABEL'] == 'France']
    df = df.drop('REF_AREA', axis=1)
    df = df.drop('REF_AREA_LABEL', axis=1)
    # print(df.head())
    # print(df.shape)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)  # S'assurer que les données sont triées

    # Features temporelles de base
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day

    # Cyclical encodings améliorés
    df["dayofyear_sin"] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365.25)

    df["month_sin"] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[date_col].dt.month / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df[date_col].dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df[date_col].dt.weekday / 7)
    
    # Features lag (valeurs précédentes de la target)
    if add_lag_features:
        for lag in [1, 2, 3, 7, 12]:  # 1 jour, 2 jours, 3 jours, 1 semaine, 1 mois
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    # Rolling statistics (moyennes mobiles)
    if add_rolling_features:
        for window in [7, 14, 30]:  # 1 semaine, 2 semaines, 1 mois
            df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std()
            df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window, min_periods=1).max()

    # Drop useless columns
    cols_to_drop = [date_col]
    if drop_cols:
        cols_to_drop += drop_cols

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
 
    # Meilleure gestion des valeurs manquantes (interpolation au lieu de fillna(0))
    # Les features lag et rolling peuvent créer des NaN au début
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='linear', limit_direction='forward')
            df[col] = df[col].fillna(df[col].median())  # Remplir les premières valeurs restantes avec la médiane

    # Split X / y AVANT le sliding window pour éviter les fuites de données
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    nb_of_samples = len(X)

    # Split temporel (chronologique)
    split_idx_test = int(nb_of_samples * split_ratio_train_test)
    X_train_full = X[:split_idx_test]
    X_test = X[split_idx_test:]
    y_train_full = y[:split_idx_test]
    y_test = y[split_idx_test:]

    nb_of_training_samples = len(X_train_full)
    split_idx_valid = int(nb_of_training_samples * split_ratio_train_valid)
    
    X_train = X_train_full[:split_idx_valid]
    X_valid = X_train_full[split_idx_valid:]
    y_train = y_train_full[:split_idx_valid]
    y_valid = y_train_full[split_idx_valid:]

    # Normalisation des features X (fit sur train uniquement)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_valid = scaler_X.transform(X_valid)
    X_test = scaler_X.transform(X_test)

    # Normalisation optionnelle de la target
    scaler_y = None
    if normalize_target:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_valid = scaler_y.transform(y_valid.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Sliding windows
    X_train, y_train = sliding_window(X_train, y_train, window_size, stride)
    X_valid, y_valid = sliding_window(X_valid, y_valid, window_size, stride)
    X_test, y_test = sliding_window(X_test, y_test, window_size, stride)

    # Passage en tenseurs et sur le device souhaité
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).to(device)

    # Reshape y si nécessaire (pour compatibilité avec le modèle)
    if len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(-1)
    if len(y_valid.shape) == 1:
        y_valid = y_valid.unsqueeze(-1)
    if len(y_test.shape) == 1:
        y_test = y_test.unsqueeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    if normalize_target:
        return train_dataset, val_dataset, test_dataset, scaler_X, scaler_y
    else:
        return train_dataset, val_dataset, test_dataset, scaler_X, None


### ------------------------------------------------------- ###


class gru(nn.Module):
    def __init__(self, input_size, mode, hidden_size=64, num_layers=2, dropout=0.2,
                 bidirectional=False, fc_hidden_size=None):
        ''' 
        GRU amélioré avec multi-layer, dropout et architecture flexible
        
        Parameters:
        input_size  - int  - number of features
        hidden_size - int  - dimension of the hidden state 
        mode        - str  - activation function: 'relu' ou 'gelu'
        num_layers  - int  - number of GRU layers (default: 2)
        dropout     - float - dropout rate between GRU layers (default: 0.2)
        bidirectional - bool - use bidirectional GRU (default: False)
        fc_hidden_size - int - size of FC hidden layer (default: hidden_size)
        '''
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if mode == "relu":
            self.activation = nn.ReLU()
        elif mode == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        # Multi-layer GRU avec dropout
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Taille de sortie du GRU (x2 si bidirectional)
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers avec dropout
        fc_hidden = fc_hidden_size if fc_hidden_size else hidden_size
        self.fc1 = nn.Linear(gru_output_size, fc_hidden)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.fc3 = nn.Linear(fc_hidden // 2, 1)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids pour améliorer la convergence"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        '''
        Forward pass amélioré

        Parameters:
        x: tensor [batch, seq_len, input_size]

        Returns:
        out: tensor [batch, 1]
        '''
        out, _ = self.gru(x)
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout_fc(out)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout_fc(out)
        
        out = self.fc3(out)

        return out


### ------------------------------------------------------- ###


class lstm(nn.Module):
    def __init__(self, input_size, mode,use_batch_norm=False, hidden_size=64, num_layers=2, dropout=0.2, 
                 bidirectional=False, fc_hidden_size=None):
        ''' 
        LSTM amélioré avec multi-layer, dropout et architecture flexible
        
        Parameters:
        input_size - int - number of features
        hidden_size - int - dimension of the hidden state 
        mode - string - activation function: 'relu' or 'gelu'
        num_layers - int - number of LSTM layers (default: 2)
        dropout - float - dropout rate between LSTM layers (default: 0.2)
        bidirectional - bool - use bidirectional LSTM (default: False)
        fc_hidden_size - int - size of FC hidden layer (default: hidden_size)
        '''
        
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if mode == "relu":
            self.activation = nn.ReLU()
        elif mode == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")
        
        # Multi-layer LSTM avec dropout
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Taille de sortie du LSTM (x2 si bidirectional)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers avec dropout
        fc_hidden = fc_hidden_size if fc_hidden_size else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden) if use_batch_norm else self.identity

        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.bn2 = nn.BatchNorm1d(fc_hidden // 2) if use_batch_norm else self.identity
        
        self.fc3 = nn.Linear(fc_hidden // 2, 1)
        
        # Initialisation des poids pour améliorer la convergence
        self._init_weights()

    def identity(self, x):
        return x
    def _init_weights(self):
        """Initialisation des poids pour améliorer la convergence"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Initialisation pour les poids d'input
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Initialisation pour les poids récurrents
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialisation des biais
                param.data.fill_(0)
                # Set forget gate bias to 1 (améliore la convergence)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            elif 'fc' in name and 'weight' in name:
                # Initialisation des couches fully connected
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        ''' 
        Forward function améliorée

        Parameters:
        x: tensor [batch, seq_len, input_size]

        Returns:
        out: tensor [batch, 1]
        '''
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Prendre la dernière sortie de la séquence
        out = lstm_out[:, -1, :]
        
        # Fully connected layers avec dropout
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout_fc(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout_fc(out)
        
        out = self.fc3(out)
        
        return out