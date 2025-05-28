import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("")

# Drop the first column (e.g., patient name)
df = df.iloc[:, 1:]

# Define target column
target_column = df.columns[-1]

# Drop rows with missing target values
df = df.dropna(subset=[target_column])

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical variables to numeric
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

### ========== RANDOM FOREST ========== ###
rf = RandomForestClassifier(
    n_estimators=75,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)

rf.fit(X_train, y_train)

# === Training Performance ===
y_train_pred_rf = rf.predict(X_train)
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
print(f"\nRandom Forest - Training Accuracy: {train_accuracy_rf:.4f}")

# === Test Performance ===
y_test_pred_rf = rf.predict(X_test)
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

# Random Forest - Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Visualize one decision tree
tree = rf.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    max_depth=5,
    feature_names=X.columns,
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    fontsize=8
)
plt.title("Random Forest - Sample Decision Tree (max_depth=5)")
plt.show()


### ========== K-NEAREST NEIGHBORS ========== ###
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test predictions
y_test_pred_knn = knn.predict(X_test)

# Metrics
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

# KNN - Confusion Matrix
cm_knn = confusion_matrix(y_test, y_test_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Oranges')
plt.title("KNN - Confusion Matrix")
plt.show()
