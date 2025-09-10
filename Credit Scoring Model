# =====================================
# TASK 1: Credit Scoring Model
# =====================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# 1️ Generate Synthetic Dataset

np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Income': np.random.normal(50000, 15000, n_samples).astype(int),  # Annual income
    'Debt': np.random.normal(15000, 5000, n_samples).astype(int),
    'CreditHistory': np.random.randint(1, 25, n_samples),  # months of credit
    'LatePayments': np.random.randint(0, 10, n_samples),
    'Age': np.random.randint(21, 65, n_samples),
    'EmploymentStatus': np.random.choice(['Employed', 'Unemployed'], n_samples, p=[0.8, 0.2]),
    'LoanAmount': np.random.randint(5000, 50000, n_samples)
})

# Target variable: 1 = creditworthy, 0 = not creditworthy
# Simple rule-based generation for demonstration
data['Approved'] = ((data['Income'] > 40000) & (data['Debt'] < 20000) & (data['LatePayments'] < 3)).astype(int)

print("Sample dataset:")
display(data.head())


# 2️ Preprocessing

# Encode categorical features
le = LabelEncoder()
data['EmploymentStatus'] = le.fit_transform(data['EmploymentStatus'])  # Employed=1, Unemployed=0

# Split features and target
X = data.drop('Approved', axis=1)
y = data['Approved']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3️ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4️ Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}


# 5️ Training & Evaluation-
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    except:
        auc = None
    
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}
    print(classification_report(y_test, y_pred))


# 6️ Summary Table
res_df = pd.DataFrame(results).T
print("\nModel Comparison:")
display(res_df)

# 7 Confusion Matrix for best model (by Accuracy)
best_model_name = res_df['Accuracy'].idxmax()
print(f"\nBest Model by Accuracy: {best_model_name}")
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
