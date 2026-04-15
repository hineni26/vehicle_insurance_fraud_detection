from xgboost import XGBClassifier

def build_model():
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=10,  # handles imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return model