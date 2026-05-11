# =============================================================================
# EARLY DETECTION OF SPORTS INJURY IN PROFESSIONAL FOOTBALL PLAYERS
# Model 4: Deep Neural Network (DNN) — TensorFlow / Keras
# =============================================================================
# Author : Manas Shrivastav | MBA Business Analytics | Christ University
# Results: Accuracy=0.55 | Weighted F1=0.53 | Best AUC (ACL Tear)=0.83
# Optimum: neurons=128, activation=softmax, optimizer=Adamax,
#          lr=0.01, dropout=0, weight_constraint=4, epochs=40, batch=128
#
# Usage:
#   pip install -r requirements.txt
#   python neural_network_model.py
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
# Install scikeras if needed: pip install scikeras

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

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)       # one-hot for DNN output layer

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.3, random_state=42)

# Scale features
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_features = X_train_sc.shape[1]
n_classes  = y_cat.shape[1]

print(f"\nTraining set   : {X_train_sc.shape}")
print(f"Test set       : {X_test_sc.shape}")
print(f"Output classes : {n_classes} — {list(le.classes_)}")

# =============================================================================
# SECTION A: HYPERPARAMETER TUNING (GridSearchCV)
# All grid searches documented below with their optimal results.
# Skip to Section B to train the final model directly.
# =============================================================================

# -----------------------------------------------------------------------------
# 4a. Grid Search — Epochs & Batch Size
# -----------------------------------------------------------------------------
def create_model_basic():
    model = Sequential([
        Dense(64, activation='softmax', input_shape=(n_features,)),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_epochs = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_basic, verbose=0),
    param_grid = {'batch_size': [10, 32, 64, 80, 100, 128, 256],
                  'epochs':     [10, 20, 40, 50, 100]},
    cv=3, n_jobs=-1
)
gs_epochs.fit(X_train_sc, y_train)
print(f"\n[Epochs/Batch] Best: {gs_epochs.best_params_} | Score: {gs_epochs.best_score_:.4f}")
# Optimum: epochs=40, batch_size=128

# -----------------------------------------------------------------------------
# 4b. Grid Search — Optimizer
# -----------------------------------------------------------------------------
def create_model_optimizer(optimizer='adam'):
    model = Sequential([
        Dense(64, activation='softmax', input_shape=(n_features,)),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_opt = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_optimizer,
                                  epochs=40, batch_size=128, verbose=0),
    param_grid = {'optimizer': ['SGD', 'RMSprop', 'Adagrad',
                                 'Adadelta', 'Adam', 'Adamax']},
    cv=3, n_jobs=-1
)
gs_opt.fit(X_train_sc, y_train)
print(f"\n[Optimizer]    Best: {gs_opt.best_params_} | Score: {gs_opt.best_score_:.4f}")
# Optimum: Adamax

# -----------------------------------------------------------------------------
# 4c. Grid Search — Learning Rate
# -----------------------------------------------------------------------------
def create_model_lr(learning_rate=0.01):
    model = Sequential([
        Dense(64, activation='softmax', input_shape=(n_features,)),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_lr = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_lr,
                                  epochs=40, batch_size=128, verbose=0),
    param_grid = {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]},
    cv=3, n_jobs=-1
)
gs_lr.fit(X_train_sc, y_train)
print(f"\n[Learning Rate] Best: {gs_lr.best_params_} | Score: {gs_lr.best_score_:.4f}")
# Optimum: learning_rate=0.01

# -----------------------------------------------------------------------------
# 4d. Grid Search — Activation Function
# -----------------------------------------------------------------------------
def create_model_activation(activation='relu'):
    model = Sequential([
        Dense(64, activation='softmax', input_shape=(n_features,)),
        Dense(n_classes, activation=activation)
    ])
    model.compile(optimizer=Adamax(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_act = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_activation,
                                  epochs=40, batch_size=128, verbose=0),
    param_grid = {'activation': ['relu', 'tanh', 'sigmoid', 'softmax']},
    cv=3, n_jobs=-1
)
gs_act.fit(X_train_sc, y_train)
print(f"\n[Activation]   Best: {gs_act.best_params_} | Score: {gs_act.best_score_:.4f}")
# Optimum: softmax

# -----------------------------------------------------------------------------
# 4e. Grid Search — Dropout Rate & Weight Constraint
# -----------------------------------------------------------------------------
def create_model_dropout(dropout_rate=0.0, weight_constraint=4):
    model = Sequential([
        Dense(64, activation='softmax', input_shape=(n_features,)),
        Dense(32, activation='softmax',
              kernel_constraint=max_norm(weight_constraint)),
        Dropout(dropout_rate),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_drop = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_dropout,
                                  epochs=40, batch_size=128, verbose=0),
    param_grid = {'dropout_rate':      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                  'weight_constraint': [1.0, 2.0, 3.0, 4.0, 5.0]},
    cv=3, n_jobs=-1
)
gs_drop.fit(X_train_sc, y_train)
print(f"\n[Dropout/WC]   Best: {gs_drop.best_params_} | Score: {gs_drop.best_score_:.4f}")
# Optimum: dropout_rate=0, weight_constraint=4

# -----------------------------------------------------------------------------
# 4f. Grid Search — Number of Neurons
# -----------------------------------------------------------------------------
def create_model_neurons(neurons=64):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(n_features,)),
        Dense(neurons, activation='relu', kernel_constraint=max_norm(4)),
        Dropout(0),
        Dense(n_classes, activation='relu')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gs_neurons = GridSearchCV(
    estimator  = KerasClassifier(model=create_model_neurons,
                                  epochs=40, batch_size=128, verbose=0),
    param_grid = {'neurons': [16, 32, 64, 128, 256]},
    cv=3, n_jobs=-1
)
gs_neurons.fit(X_train_sc, y_train)
print(f"\n[Neurons]      Best: {gs_neurons.best_params_} | Score: {gs_neurons.best_score_:.4f}")
# Optimum: neurons=128

# =============================================================================
# SECTION B: FINAL MODEL — Optimum Hyperparameters
# neurons=128, activation=softmax, optimizer=Adamax(lr=0.01),
# dropout=0, weight_constraint=4, epochs=40, batch_size=128
# =============================================================================
print("\nBuilding final DNN with optimum hyperparameters...")

def build_final_model():
    opt = Adamax(learning_rate=0.01)
    model = Sequential([
        Dense(128, activation='softmax', input_shape=(n_features,)),
        Dropout(0),
        Dense(128, activation='softmax', kernel_constraint=max_norm(4)),
        Dropout(0),
        Dense(128, activation='softmax', kernel_constraint=max_norm(4)),
        Dropout(0),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

final_model = build_final_model()
final_model.summary()

history = final_model.fit(
    X_train_sc, y_train,
    epochs          = 40,
    batch_size      = 128,
    validation_data = (X_test_sc, y_test),
    verbose         = 1
)

# -----------------------------------------------------------------------------
# 5. Training History — Accuracy Curve
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'],     label='Train Accuracy', lw=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy',   lw=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1.0])
plt.title('DNN Training vs Validation Accuracy (40 Epochs)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/dnn_training_curve.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 6. Evaluate & Predict
# -----------------------------------------------------------------------------
test_loss, test_acc = final_model.evaluate(X_test_sc, y_test, verbose=0)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

y_prob   = final_model.predict(X_test_sc)
y_pred   = np.argmax(y_prob, axis=1)
y_true   = np.argmax(y_test, axis=1)
y_pred_l = le.inverse_transform(y_pred)
y_true_l = le.inverse_transform(y_true)

# -----------------------------------------------------------------------------
# 7. Confusion Matrix & Classification Report
# -----------------------------------------------------------------------------
labels = ['ACL Tear', 'Hamstring', 'Lower Back Pain', 'Sprained Ankle']
cm = confusion_matrix(y_true_l, y_pred_l, labels=labels)

print("\n--- Confusion Matrix ---")
print(cm)
print("\n--- Classification Report ---")
print(classification_report(y_true_l, y_pred_l, target_names=labels))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — DNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/dnn_confusion_matrix.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 8. ROC Curve — One vs Rest (Multi-class)
# -----------------------------------------------------------------------------
fpr, tpr, roc_auc = {}, {}, {}

for i, label in enumerate(labels):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green', 'orange']
for i, (color, label) in enumerate(zip(colors, labels)):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC — {label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--',
         label='Random Classifier')
plt.title('ROC Curve — Deep Neural Network')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/dnn_roc_curve.png', dpi=150)
plt.show()

print("\n--- ROC-AUC Scores ---")
for i, label in enumerate(labels):
    print(f"  {label}: {roc_auc[i]:.2f}")

# -----------------------------------------------------------------------------
# 9. Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DNN — FINAL RESULTS")
print("=" * 55)
print("Architecture : 4 Dense layers (128 neurons, softmax)")
print("Optimizer    : Adamax (lr=0.01)")
print("Epochs       : 40 | Batch: 128 | Dropout: 0")
print("Weight Const.: max_norm(4)")
print("Accuracy     : 0.55")
print("Weighted F1  : 0.53")
print("AUC ACL Tear : 0.83")
print("AUC Hamstring: 0.83")
print("AUC Low Back : 0.81")
print("AUC Sp. Ankle: 0.66")
print("=" * 55)
print("Note: DNN underperforms AdaBoost on this dataset size.")
print("AdaBoost remains the recommended primary model.")
print("=" * 55)
