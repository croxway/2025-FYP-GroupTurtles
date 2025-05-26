
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

# === Load dataset ===
file_path = " "  # ← Replace this with your actual file path
df = pd.read_csv(file_path, delimiter=';')

# ------------------- RANDOM FOREST PIPELINE -------------------
print("\n========== RANDOM FOREST ==========\n")

# Drop missing labels
df_rf = df.dropna(subset=['label'])

# Features and target
X_rf = df_rf.drop(columns=['label', 'key'], errors='ignore')
y_rf = df_rf['label'].astype(int)

# Identify column types
cat_cols = X_rf.select_dtypes(include='object').columns.tolist()
num_cols = X_rf.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor_rf = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Build pipeline
pipeline_rf = ImbPipeline(steps=[
    ('preprocessor', preprocessor_rf),
    ('resample', SMOTETomek(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    ))
])

# Train-test split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, stratify=y_rf, random_state=42
)

# Fit Random Forest
pipeline_rf.fit(X_train_rf, y_train_rf)

# Predict & Evaluate
y_pred_rf = pipeline_rf.predict(X_test_rf)
y_proba_rf = pipeline_rf.predict_proba(X_test_rf)[:, 1]

print("Confusion Matrix (Random Forest):")
ConfusionMatrixDisplay.from_predictions(y_test_rf, y_pred_rf)
plt.show()

target_names = ["(Benign) 0", "(Malignant) 1"]
print("\nClassification Report (Random Forest):")
print(classification_report(y_test_rf, y_pred_rf, target_names=target_names, digits=3))

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test_rf, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf, roc_auc=roc_auc_rf).plot()
plt.title("ROC Curve - Random Forest")
plt.grid(True)
plt.show()

# Precision-Recall
precision_rf, recall_rf, _ = precision_recall_curve(y_test_rf, y_proba_rf)
PrecisionRecallDisplay(precision=precision_rf, recall=recall_rf).plot()
plt.title("Precision-Recall Curve - Random Forest")
plt.grid(True)
plt.show()

# Visualize a tree
estimator = pipeline_rf.named_steps['classifier'].estimators_[0]
preprocessor_fit = pipeline_rf.named_steps['preprocessor']
onehot_features = preprocessor_fit.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_features = num_cols + list(onehot_features)

plt.figure(figsize=(24, 16))
plot_tree(
    estimator,
    feature_names=all_features,
    class_names=["Benign (0)", "Malignant (1)"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=None
)
plt.title("Single Decision Tree from Random Forest", fontsize=16)
plt.show()

# ------------------- KNN PIPELINE -------------------
print("\n========== K-NEAREST NEIGHBORS (KNN) ==========\n")

# Drop non-numeric columns
excluded_cols = ['key', 'label', 'Diagnosis'] + [f'Color {i}' for i in range(1, 6)]
X_knn = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
X_knn = X_knn.apply(pd.to_numeric, errors='coerce')

# Filter labeled rows
labeled_mask = df['label'].notnull()
X_knn_labeled = X_knn[labeled_mask]
y_knn_labeled = df.loc[labeled_mask, 'label'].astype(int)

# Train-test split
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn_labeled, y_knn_labeled, test_size=0.2, stratify=y_knn_labeled, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_knn_scaled = scaler.fit_transform(X_train_knn)
X_test_knn_scaled = scaler.transform(X_test_knn)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn_scaled, y_train_knn)

# Predict & Evaluate
y_pred_knn = knn.predict(X_test_knn_scaled)

print("Confusion Matrix (KNN):")
ConfusionMatrixDisplay.from_predictions(y_test_knn, y_pred_knn)
plt.show()

print("\nClassification Report (KNN):")
print(classification_report(y_test_knn, y_pred_knn, target_names=target_names, digits=3))
