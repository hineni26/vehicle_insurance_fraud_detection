from preprocessing import load_data, preprocess_data, split_data, reshape_for_lstm, normalize_data
from model import build_model
import tensorflow as tf
from tensorflow import keras

def train_pipeline(data_path, target_column, model_type='xgboost'):
    """
    Training pipeline supporting both XGBoost and LSTM models
    
    Args:
        data_path (str): Path to CSV file
        target_column (str): Name of target column
        model_type (str): 'xgboost' or 'lstm'
    
    Returns:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    # Data loading and preprocessing
    df = load_data(data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    if model_type == 'xgboost':
        return _train_xgboost(X_train, X_test, y_train, y_test)
    elif model_type == 'lstm':
        return _train_lstm(X_train, X_test, y_train, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def _train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    model = build_model(model_type='xgboost')
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def _train_lstm(X_train, X_test, y_train, y_test):
    """Train LSTM model"""
    # Normalize data
    X_train_scaled, X_test_scaled = normalize_data(X_train.values, X_test.values)
    
    # Reshape for LSTM
    X_train_lstm = reshape_for_lstm(X_train_scaled, timesteps=8)
    X_test_lstm = reshape_for_lstm(X_test_scaled, timesteps=8)
    
    # Build and train model
    model = build_model(model_type='lstm', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
    
    # Train with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, X_test_lstm, y_test