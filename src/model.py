from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(model_type='xgboost', input_shape=None):
    """
    Build either XGBoost or LSTM model
    
    Args:
        model_type (str): 'xgboost' or 'lstm'
        input_shape (tuple): Required for LSTM. Shape of input data (timesteps, features)
    
    Returns:
        model: Trained model
    """
    if model_type == 'xgboost':
        return build_xgboost_model()
    elif model_type == 'lstm':
        if input_shape is None:
            raise ValueError("input_shape is required for LSTM model")
        return build_lstm_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_xgboost_model():
    """Build XGBoost classifier"""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=10,  # handles imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return model

def build_lstm_model(input_shape):
    """
    Build LSTM neural network for fraud detection
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
    
    Returns:
        model: Compiled Keras LSTM model
    """
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model