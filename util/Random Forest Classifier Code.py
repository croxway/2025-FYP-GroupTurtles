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
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.tree import plot_tree

# Load dataset
file_path = r"C:\Users\suraj\Downloads\2025-FYP-Turtles\Baseline_classification.csv"
df = pd.read_csv(file_path)

# Drop missing labels
df = df.dropna(subset=['label'])

# Features and target
X = df.drop(columns=['label', 'key'])
y = df['label']

# Identify columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline with SMOTETomek and Random Forest
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', SMOTETomek(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fit model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Confusion matrix
print("\nConfusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Classification report with custom label formatting
class_mapping = {0: "Benign", 1: "Malignant"}
target_names_custom = [f"({class_mapping[i]}) {i}" for i in sorted(class_mapping.keys())]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names_custom, digits=3))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title("ROC Curve")
plt.grid(True)
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

# Get one tree from the Random Forest
estimator = pipeline.named_steps['classifier'].estimators_[0]

# Get actual feature names from preprocessing
preprocessor_fit = pipeline.named_steps['preprocessor']
onehot_features = preprocessor_fit.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_features = num_cols + list(onehot_features)

# Plot the decision tree
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
