"""
Main script to train and evaluate both XGBoost and LSTM models
for vehicle insurance fraud detection
"""

from train import train_pipeline
from evaluate import evaluate_model

def main():
    data_path = '../data/sample_data.csv'
    target_column = 'FraudFound_P'
    
    print("\n" + "="*60)
    print("VEHICLE INSURANCE FRAUD DETECTION")
    print("="*60)
    
    # Train and evaluate XGBoost model
    print("\n\n### TRAINING XGBOOST MODEL ###\n")
    xgb_model, X_test_xgb, y_test_xgb = train_pipeline(
        data_path, 
        target_column, 
        model_type='xgboost'
    )
    evaluate_model(xgb_model, X_test_xgb, y_test_xgb, model_type='xgboost')
    
    # Train and evaluate LSTM model
    print("\n\n### TRAINING LSTM MODEL ###\n")
    lstm_model, X_test_lstm, y_test_lstm = train_pipeline(
        data_path, 
        target_column, 
        model_type='lstm'
    )
    evaluate_model(lstm_model, X_test_lstm, y_test_lstm, model_type='lstm')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
