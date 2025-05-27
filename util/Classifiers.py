import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /2025-FYP-Turtles/Baseline_classification.csv")
# Drop rows with missing label
df = df.dropna(subset=['label'])

# Separate features and label
x = df.drop(columns=['key', 'label'])
y = df['label'].astype(int)

# Scale features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Split the dataset (still good for final test set and tree visualization)
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)


# --- Random Forest Classifier with Cross-Validation & Grid Search ---
print("--- Random Forest Training & Evaluation ---")
rf_param_grid = {
    'n_estimators': [100, 200, 300], # Fewer options for faster example
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'class_weight': ['balanced']
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5, # 5-fold cross-validation
    scoring='f1', # Optimize for F1-score, given class_weight='balanced'
    n_jobs=-1, # Use all available CPU cores
    verbose=1
)
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1 Score': f1_score(y_test, y_pred_rf)
}

print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")


# --- KNN Classifier with Cross-Validation & Basic Hyperparameter Tuning ---
print("\n--- KNN Training & Evaluation ---")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'] # Add weights parameter
}

knn_grid_search = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
knn_grid_search.fit(X_train, y_train)

best_knn = knn_grid_search.best_estimator_
y_pred_knn = best_knn.predict(X_test)

knn_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_knn),
    'Precision': precision_score(y_test, y_pred_knn),
    'Recall': recall_score(y_test, y_pred_knn),
    'F1 Score': f1_score(y_test, y_pred_knn)
}

print(f"Best KNN Parameters: {knn_grid_search.best_params_}")


# --- Decision Tree (depth=2) ---
clf_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
clf_tree.fit(X_train, y_train)

# --- Plotting Confusion Matrices ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(ax=axs[0], cmap='Blues')
axs[0].set_title("Random Forest Confusion Matrix")

# KNN confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn)).plot(ax=axs[1], cmap='Oranges')
axs[1].set_title("KNN Confusion Matrix")

plt.tight_layout()
plt.show()

# --- Separate Plot: Decision Tree ---
plt.figure(figsize=(14, 8))
tree.plot_tree(
    clf_tree,
    max_depth=2,
    feature_names=x.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)
plt.title("Decision Tree (depth=2)")
plt.show()

# --- Print Performance Metrics ---
print("\n=== Random Forest Metrics (on test set) ===")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.2f}")

print("\n=== KNN Metrics (on test set) ===")
for k, v in knn_metrics.items():
    print(f"{k}: {v:.2f}")