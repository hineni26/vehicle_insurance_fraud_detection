from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np

def evaluate_model(model, X_test, y_test, model_type='xgboost'):
    """
    Evaluate model performance
    
    Args:
        model: Trained model (XGBoost or LSTM)
        X_test: Test features
        y_test: Test labels
        model_type (str): 'xgboost' or 'lstm'
    """
    if model_type == 'xgboost':
        _evaluate_xgboost(model, X_test, y_test)
    elif model_type == 'lstm':
        _evaluate_lstm(model, X_test, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def _evaluate_xgboost(model, X_test, y_test):
    """Evaluate XGBoost model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("=" * 50)
    print("XGBOOST MODEL EVALUATION")
    print("=" * 50)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

def _evaluate_lstm(model, X_test, y_test):
    """Evaluate LSTM model"""
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("=" * 50)
    print("LSTM MODEL EVALUATION")
    print("=" * 50)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))