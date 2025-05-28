import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
)
from sklearn.tree import plot_tree


df = pd.read_csv('')
df = df.iloc[:, 1:]  
target_column = df.columns[-1]
df = df.dropna(subset=[target_column])  


X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [4, 6, 8],
    'min_samples_split': [4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)
print(f"Cross-validated best accuracy: {grid_search.best_score_:.4f}")

# === Train Performance ===
y_train_pred_rf = best_rf.predict(X_train)
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
print(f"\nRandom Forest - Training Accuracy: {train_accuracy_rf:.4f}")

# === Test Performance ===
y_test_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
precision_rf = precision_score(y_test, y_test_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_test_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_test_pred_rf, average='weighted', zero_division=0)

print(f"\nRandom Forest - Test Accuracy:  {accuracy_rf:.4f}")
print(f"Random Forest - Test Precision: {precision_rf:.4f}")
print(f"Random Forest - Test Recall:    {recall_rf:.4f}")
print(f"Random Forest - Test F1 Score:  {f1_rf:.4f}")
print("\nRandom Forest - Classification Report:\n")
print(classification_report(y_test, y_test_pred_rf, zero_division=0))


ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.show()


importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), features[:10], rotation=45, ha='right')
plt.tight_layout()
plt.show()


y_prob_rf = best_rf.predict_proba(X_test)[:, 0]  
fpr, tpr, _ = roc_curve(y_test, y_prob_rf, pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Random Forest").plot()
plt.title("Random Forest - ROC Curve (Class 0 as Positive)")
plt.show()


precision, recall, _ = precision_recall_curve(y_test, y_prob_rf, pos_label=0)

plt.figure()
PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name="Random Forest").plot()
plt.title("Random Forest - Precision-Recall Curve (Class 0 as Positive)")
plt.show()


plt.figure(figsize=(20, 10))
plot_tree(best_rf.estimators_[0], 
          feature_names=X.columns, 
          class_names=[str(cls) for cls in np.unique(y)],
          filled=True, max_depth=2, fontsize=10)
plt.title("Decision Tree from Random Forest (Depth=2)")
plt.show()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_test_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
precision_knn = precision_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)
recall_knn = recall_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)
f1_knn = f1_score(y_test, y_test_pred_knn, average='weighted', zero_division=0)

print(f"\nKNN - Test Accuracy:  {accuracy_knn:.4f}")
print(f"KNN - Test Precision: {precision_knn:.4f}")
print(f"KNN - Test Recall:    {recall_knn:.4f}")
print(f"KNN - Test F1 Score:  {f1_knn:.4f}")
print("\nKNN - Classification Report:\n")
print(classification_report(y_test, y_test_pred_knn, zero_division=0))


ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, cmap='Oranges')
plt.title("KNN - Confusion Matrix")
plt.show()
