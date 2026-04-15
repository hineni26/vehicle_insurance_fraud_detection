# 🚗 Vehicle Insurance Fraud Detection using XGBoost

## 📌 Overview

Insurance fraud leads to significant financial losses for companies every year. Detecting fraudulent claims manually is inefficient due to large-scale data and hidden patterns.

This project builds a **machine learning-based fraud detection system** using **XGBoost**, focusing on handling class imbalance and improving fraud detection performance.

---

## 📊 Dataset

* ~15,000+ insurance claim records
* 33 features including:

  * Customer details (Age, PolicyType)
  * Vehicle information (VehicleCategory, VehiclePrice)
  * Claim-related attributes

### 🎯 Target Variable

* `FraudFound_P`

  * `0` → Legitimate claim
  * `1` → Fraudulent claim

### ⚠️ Key Challenge

* Dataset is **imbalanced** (fraud cases are rare)

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* Google Colab

---

## 🔍 Workflow

1. Data Cleaning & Preprocessing
2. Encoding categorical features
3. Handling class imbalance
4. Train-test split with stratification
5. Model training using XGBoost
6. Model evaluation using multiple metrics
7. Feature importance analysis

---

## 🧠 Model Used

* **XGBoost Classifier**

  * Handles non-linear relationships
  * Robust to imbalanced datasets using `scale_pos_weight`
  * Provides feature importance insights

---

## 📊 Evaluation Metrics

Since fraud detection is an imbalanced problem, accuracy alone is not sufficient.

We use:

* **Precision** → Correct fraud predictions
* **Recall** → Ability to detect actual fraud cases
* **ROC-AUC Score** → Overall model performance
* **Confusion Matrix** → Error analysis

---

## 📈 Results

* High recall achieved for fraud detection
* Model successfully identifies fraud patterns
* Feature importance highlights key contributing factors

---

## 🔥 Key Insights

* Fraud detection is a **cost-sensitive problem**
* Missing fraud (false negatives) is more critical than false alarms
* Certain features contribute significantly to fraud prediction

---

## 📁 Project Structure

```
vehicle-insurance-fraud-detection/
│
├── data/
│   └── sample_data.csv
│
├── notebooks/
│   └── xgboost_fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
├── dataset_info.md
└── README.md
```

---

## 🚀 How to Run

### 1. Clone Repository

```
git clone https://github.com/SoumyasreeMitra/Vehicle-Insurance-Fraud-Detection.git
cd vehicle-insurance-fraud-detection
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Project

```
python src/main.py
```

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV / Optuna)
* Use advanced models like LightGBM
* Deploy using Streamlit or Flask
* Real-time fraud detection system

---

## 📌 Conclusion

This project demonstrates how machine learning, especially XGBoost, can effectively detect fraudulent insurance claims while handling real-world challenges like class imbalance.

---

## 👩‍💻 Author

Soumyasree Mitra
