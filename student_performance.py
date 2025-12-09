from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from training_functions import activation_function
import numpy as np
from torch.utils.data import TensorDataset

def student_preprocessing(
    df: pd.DataFrame, 
    val_size, 
    test_size, 
    add_interactions: bool = True,
    add_polynomial_features: bool = False,
    polynomial_degree: int = 2,
    random_state: int = 1
) -> tuple:
    """
    Preprocessing amélioré pour le dataset student performance.
    
    Args:
        df: DataFrame avec les données
        val_size: Proportion pour la validation
        test_size: Proportion pour le test
        add_interactions: Si True, ajoute des features d'interaction entre variables importantes
        add_polynomial_features: Si True, ajoute des features polynomiales
        polynomial_degree: Degré des features polynomiales
        random_state: Seed pour la reproductibilité
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler_X, scaler_y
    """
    df = df.copy()

    # Détection des colonnes catégorielles (object ou category)
    categorical_cols = [col for col in df.columns 
                       if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    
    # One-hot encoding pour les colonnes catégorielles
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=categorical_cols, inplace=True)
     
    # Feature engineering : interactions entre variables importantes
    if add_interactions:
        # Variables numériques importantes pour les interactions
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Exam_Score' in numerical_cols:
            numerical_cols.remove('Exam_Score')
        
        # Interactions entre variables clés (si elles existent)
        key_vars = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours']
        available_key_vars = [v for v in key_vars if v in numerical_cols]
        
        if len(available_key_vars) >= 2:
            # Interactions entre les 2 premières variables clés
            if len(available_key_vars) >= 2:
                df[f'{available_key_vars[0]}_x_{available_key_vars[1]}'] = (
                    df[available_key_vars[0]] * df[available_key_vars[1]]
                )
            # Ratio entre variables importantes
            if 'Hours_Studied' in available_key_vars and 'Sleep_Hours' in available_key_vars:
                df['Study_Sleep_Ratio'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 1e-6)
            if 'Previous_Scores' in available_key_vars and 'Attendance' in available_key_vars:
                df['Score_Attendance_Ratio'] = df['Previous_Scores'] / (df['Attendance'] + 1e-6)
    
    # Features polynomiales optionnelles
    if add_polynomial_features:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Exam_Score' in numerical_cols:
            numerical_cols.remove('Exam_Score')
        
        # Ajouter des features polynomiales pour les variables les plus importantes
        important_vars = ['Hours_Studied', 'Attendance', 'Previous_Scores']
        available_important = [v for v in important_vars if v in numerical_cols]
        
        for var in available_important[:3]:  # Limiter à 3 variables pour éviter l'explosion
            for degree in range(2, polynomial_degree + 1):
                df[f'{var}_pow_{degree}'] = df[var] ** degree

    # Séparation features/target
    X_student = df.drop('Exam_Score', axis=1)
    y_student = df['Exam_Score']
    
    assert val_size + test_size < 1

    # Split train/val/test
    val_size_adjusted = val_size / (1 - test_size)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_student, y_student, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    # Scaling des features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scaling de la target (optionnel mais utile pour la normalisation)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()


    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
    
    
    return train_dataset, val_dataset, test_dataset, scaler_X, scaler_y


class student_model(nn.Module):
    def __init__(
        self, 
        input_dim, 
        mode='gelu',
        hidden_dims=[64, 32],
        dropout_rate=0.4,
        use_batch_norm=False
    ):
        """
        Modèle régularisé pour éviter l'overfitting.
        
        Args:
            input_dim: Dimension des features d'entrée
            mode: Fonction d'activation ('relu', 'gelu', 'swish', etc.)
            hidden_dims: Liste des dimensions des couches cachées (par défaut [64, 32] pour éviter l'overfitting)
            dropout_rate: Taux de dropout pour la régularisation (par défaut 0.5 pour forte régularisation)
            use_batch_norm: Si True, utilise BatchNorm (désactivé par défaut car peut interférer avec dropout élevé)
        """
        super(student_model, self).__init__()
        self.activation = activation_function(mode)
        self.mode = mode
        self.use_batch_norm = use_batch_norm
        
        # Construction des couches avec régularisation forte
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # BatchNorm seulement si demandé et avec dropout modéré
            if use_batch_norm and dropout_rate <= 0.4:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            # Dropout sur toutes les couches cachées pour régularisation forte
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)
