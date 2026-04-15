from preprocessing import load_data, preprocess_data, split_data
from model import build_model

def train_pipeline(data_path, target_column):
    df = load_data(data_path)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    model = build_model()
    model.fit(X_train, y_train)

    return model, X_test, y_test