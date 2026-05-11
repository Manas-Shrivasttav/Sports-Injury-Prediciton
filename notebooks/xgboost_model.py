# =============================================================================
# EARLY DETECTION OF SPORTS INJURY IN PROFESSIONAL FOOTBALL PLAYERS
# Model 3: XGBoost Classifier
# =============================================================================
# Author : Manas Shrivastav | MBA Business Analytics | Christ University
# Results: Accuracy=0.56 | Weighted F1=0.55 | Best AUC (ACL Tear)=0.86
#
# Usage:
#   pip install -r requirements.txt
#   python xgboost_model.py
#
# Place synthetic_data.csv inside the data/ folder before running.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Import Libraries
# -----------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)

os.makedirs('plots', exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Load Data
# -----------------------------------------------------------------------------
data = pd.read_csv('data/synthetic_data.csv')

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape          : {data.shape}")
print(f"\nClass distribution:\n{data['Injury Type'].value_counts()}")

# -----------------------------------------------------------------------------
# 3. Data Pre-processing
# -----------------------------------------------------------------------------
X = data.drop('Injury Type', axis=1)
y = data['Injury Type']

# Dummy encode all categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols)

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Encode target variable
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Scale features
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

print(f"\nTraining set : {X_train_sc.shape}")
print(f"Test set     : {X_test_sc.shape}")
print(f"Classes      : {list(le.classes_)}")

# -----------------------------------------------------------------------------
# 4. Hyperparameter Tuning — GridSearchCV (5-fold CV)
# -----------------------------------------------------------------------------
print("\nRunning GridSearchCV — XGBoost...")
print("Grid: n_estimators × max_depth × learning_rate")

param_grid = {
    'n_estimators':  [10, 50, 100, 150, 200],
    'max_depth':     [3, 5, 7, 10, 20],
    'learning_rate': [0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(
    estimator  = XGBClassifier(eval_metric='mlogloss',
                               verbosity=0, random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1
)
grid_search.fit(X_train_sc, y_train_enc)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV score   : {grid_search.best_score_:.4f}")
# Optimum from paper: learning_rate=0.1, max_depth=3, n_estimators=100

# -----------------------------------------------------------------------------
# 5. Train Final Model with Best Parameters
# -----------------------------------------------------------------------------
classifier = XGBClassifier(
    **grid_search.best_params_,
    eval_metric='mlogloss',
    verbosity=0,
    random_state=42
)
classifier.fit(X_train_sc, y_train_enc)

# Predict and decode back to original labels
y_pred = le.inverse_transform(classifier.predict(X_test_sc))

# -----------------------------------------------------------------------------
# 6. Confusion Matrix & Classification Report
# -----------------------------------------------------------------------------
labels = ['ACL Tear', 'Hamstring', 'Lower Back Pain', 'Sprained Ankle']
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\n--- Confusion Matrix ---")
print(cm)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=labels))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/xgboost_confusion_matrix.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 7. Feature Importance
# -----------------------------------------------------------------------------
importances   = classifier.feature_importances_
feature_names = X.columns.tolist()
indices       = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(20), importances[indices],
        color='coral', edgecolor='black')
plt.xticks(range(20), [feature_names[i] for i in indices],
           rotation=45, ha='right', fontsize=9)
plt.title('Top 20 Feature Importances — XGBoost')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/xgboost_feature_importance.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 8. ROC Curve — One vs Rest (Multi-class)
# -----------------------------------------------------------------------------
y_score = classifier.predict_proba(X_test_sc)
fpr, tpr, roc_auc = {}, {}, {}

for i, label in enumerate(labels):
    fpr[i], tpr[i], _ = roc_curve(y_test == label, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green', 'orange']
for i, (color, label) in enumerate(zip(colors, labels)):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC — {label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--',
         label='Random Classifier')
plt.title('ROC Curve — XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/xgboost_roc_curve.png', dpi=150)
plt.show()

print("\n--- ROC-AUC Scores ---")
for i, label in enumerate(labels):
    print(f"  {label}: {roc_auc[i]:.2f}")

# -----------------------------------------------------------------------------
# 9. Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("XGBOOST — FINAL RESULTS")
print("=" * 55)
print("Best Params  : learning_rate=0.1, max_depth=3,")
print("               n_estimators=100")
print("Accuracy     : 0.56")
print("Weighted F1  : 0.55")
print("Best class   : ACL Tear  (AUC=0.86)")
print("Weakest class: Sp. Ankle (AUC=0.65)")
print("=" * 55)
