import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load and preprocess dataset
data = pd.read_csv('creditcard.csv')
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Prediction function for Django
def predict_transaction(model_type, transaction_values):
    """
    model_type: 'lr' or 'dt'
    transaction_values: list of 30 features
    """
    transaction = np.array([transaction_values])
    transaction = scaler.transform(transaction)
    
    if model_type == 'lr':
        result = lr_model.predict(transaction)
    else:
        result = dt_model.predict(transaction)
    
    return "❌ Fraudulent Transaction" if result[0] == 1 else "✅ Genuine Transaction"
