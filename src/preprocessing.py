import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

def split_data(df, target_column, test_size=0.2):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def reshape_for_lstm(X, timesteps=None):
    """
    Reshape tabular data into sequences for LSTM
    
    Args:
        X (array-like): Input features of shape (n_samples, n_features)
        timesteps (int): Number of timesteps. If None, uses n_features as timesteps
    
    Returns:
        array: Reshaped data of shape (n_samples, timesteps, features_per_step)
    """
    n_samples, n_features = X.shape
    
    if timesteps is None:
        timesteps = n_features
    
    # Reshape features into sequences
    # Each sample becomes a sequence of timesteps
    features_per_step = max(1, n_features // timesteps)
    
    # Pad features if necessary
    padded_features = n_features
    if n_features % timesteps != 0:
        padded_features = timesteps * features_per_step
        X_padded = np.zeros((n_samples, padded_features))
        X_padded[:, :n_features] = X
        X = X_padded
    
    # Reshape to (n_samples, timesteps, features_per_step)
    X_reshaped = X.reshape(n_samples, timesteps, features_per_step)
    
    return X_reshaped

def normalize_data(X_train, X_test):
    """
    Normalize features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        X_train_scaled, X_test_scaled: Scaled features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled