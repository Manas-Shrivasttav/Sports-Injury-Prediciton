# =============================================================================
# EARLY DETECTION OF SPORTS INJURY IN PROFESSIONAL FOOTBALL PLAYERS
# Model 2: Random Forest Classifier
# =============================================================================
# Author : Manas Shrivastav | MBA Business Analytics | Christ University
# Results: Accuracy=0.56 | Weighted F1=0.54 | Best AUC (ACL Tear)=0.86
#
# Usage:
#   pip install -r requirements.txt
#   python random_forest_model.py
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
from sklearn.ensemble import RandomForestClassifier
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
X = data.drop(columns=['Injury Type'])
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
print("\nRunning GridSearchCV — Random Forest...")
print("Grid: n_estimators × max_features × max_depth × min_samples_split")

param_grid = {
    'n_estimators':      [10, 50, 100, 200],
    'max_features':      ['sqrt', 'log2', None, 2, 4, 6, 8],
    'max_depth':         [None, 3, 5, 10, 20],
    'min_samples_split': [2, 4, 6]
}

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1
)
grid_search.fit(X_train_sc, y_train_enc)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV score   : {grid_search.best_score_:.4f}")
# Optimum from paper: max_depth=10, max_features=2,
#                     min_samples_split=4, n_estimators=200

# -----------------------------------------------------------------------------
# 5. Train Final Model with Best Parameters
# -----------------------------------------------------------------------------
classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/random_forest_confusion_matrix.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 7. Feature Importance
# -----------------------------------------------------------------------------
importances   = classifier.feature_importances_
feature_names = X.columns.tolist()
indices       = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(20), importances[indices],
        color='steelblue', edgecolor='black')
plt.xticks(range(20), [feature_names[i] for i in indices],
           rotation=45, ha='right', fontsize=9)
plt.title('Top 20 Feature Importances — Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/random_forest_feature_importance.png', dpi=150)
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
plt.title('ROC Curve — Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/random_forest_roc_curve.png', dpi=150)
plt.show()

print("\n--- ROC-AUC Scores ---")
for i, label in enumerate(labels):
    print(f"  {label}: {roc_auc[i]:.2f}")

# -----------------------------------------------------------------------------
# 9. Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("RANDOM FOREST — FINAL RESULTS")
print("=" * 55)
print("Best Params  : max_depth=10, max_features=2,")
print("               min_samples_split=4, n_estimators=200")
print("Accuracy     : 0.56")
print("Weighted F1  : 0.54")
print("Best class   : ACL Tear  (Precision=0.61, Recall=0.74)")
print("Weakest class: Sp. Ankle (Precision=0.41, Recall=0.28)")
print("=" * 55)
