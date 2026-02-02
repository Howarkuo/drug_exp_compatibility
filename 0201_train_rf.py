import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SVMSMOTE
import joblib

# 1. Load your processed data
# This file contains the 200 mol2vec descriptors you generated
INPUT_FILE = '2024_Drug_compatibility_dataset.xlsx_train_data_vectors_final.csv'
print(f" Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# 2. Define X and y
# Dropping Metadata (CIDs) and Label to isolate features
X = df.drop(columns=['API_CID', 'Excipient_CID', 'Label']).values
y = df['Label'].values

print(f"   Features Shape: {X.shape}")
print(f"   Target Shape: {y.shape}")

# 3. Data Splitting (Paper: 60% Train, 20% Val, 20% Test)
print("🔪 Splitting Data (60-20-20)...")
# First split: 60% Train, 40% Remain
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
# Second split: Split the remaining 40% into half (20% Val, 20% Test)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42, stratify=y_remain)

print(f"   Train Set: {X_train.shape}")
print(f"   Val Set:   {X_val.shape}")
print(f"   Test Set:  {X_test.shape}")

# 4. Apply SVM-SMOTE (Paper: Section 2.2)
# Only apply this to the Training set to avoid data leakage
print("  Applying SVM-SMOTE to Training Set...")
try:
    svmsmote = SVMSMOTE(random_state=42, m_neighbors=10, sampling_strategy='auto',k_neighbors =5 ,out_step=0.5)
    X_train_res, y_train_res = svmsmote.fit_resample(X_train, y_train)
    print(f"   Original Incompatible count: {sum(y_train==1)}")
    print(f"   Resampled Incompatible count: {sum(y_train_res==1)}")
except Exception as e:
    print(f"SVM-SMOTE failed (likely too few minority samples for SVM neighbors). Falling back to Standard SMOTE. Error: {e}")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Model Building - Random Forest (Paper: Section 2.3)
print("\n Training Random Forest with Grid Search...")

# Parameter grid derived from your model_selection.py
# for size and complexity: n_estimators: The number of trees, max_depth: How deep each tree can grow, max_features: How many features to look at when splitting a node.
# for regularization (prevent overfitting):min_samples_split: The minimum number of samples required to split a node.min_samples_leaf: The minimum number of samples required to be at a leaf node.
# advance imbalance tuning: bootstrap: Whether to use bootstrap samples (random sampling with replacement) when building trees.class_weight: Handles class imbalance.
param_grid_RF = {
    # number of trees (combination of decision): can increase to 300 ~ 500
    'n_estimators': [100, 200],
    # Max depth of trees: Controls complexity.
    'max_depth': [10, 20, None],
    'class_weight': ['balanced', None] 
}

grid_search_RF = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid=param_grid_RF, 
    scoring='accuracy',
    cv=5, 
    n_jobs=-1, 
    verbose=1
)

grid_search_RF.fit(X_train_res, y_train_res)
best_rf = grid_search_RF.best_estimator_

print(f"   Best Parameters: {grid_search_RF.best_params_}")

# 6. Evaluation on Validation Set
print("\n Validation Set Results:")
y_pred_val = best_rf.predict(X_val)
print(classification_report(y_val, y_pred_val))
print("Confusion Matrix (Validation):")
print(confusion_matrix(y_val, y_pred_val))

# 7. Evaluation on Test Set (Final Check)
print("\n Test Set Results:")
y_pred_test = best_rf.predict(X_test)

# Explicitly calculate metrics
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average='binary')
test_cm = confusion_matrix(y_test, y_pred_test)

print(f"   Accuracy: {test_acc:.4f}")
print(f"   F1 Score (Incompatible Class): {test_f1:.4f}")
print("   Confusion Matrix:")
print(test_cm)
print("\n   Full Classification Report:")
print(classification_report(y_test, y_pred_test))
# 8. Save Model
joblib.dump(best_rf, '0201_my_rf_model.pkl')
print("\n Model saved as '0201_my_rf_model.pkl'")



# Loading 2024_Drug_compatibility_dataset.xlsx_train_data_vectors_final.csv...
#    Features Shape: (3544, 200)
#    Target Shape: (3544,)
# 🔪 Splitting Data (60-20-20)...
#    Train Set: (2126, 200)
#    Val Set:   (709, 200)
#    Test Set:  (709, 200)
#   Applying SVM-SMOTE to Training Set...
#    Original Incompatible count: 206
#    Resampled Incompatible count: 1920

#  Training Random Forest with Grid Search...
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
#    Best Parameters: {'class_weight': 'balanced', 'max_depth': 20, 'n_estimators': 200}

#  Validation Set Results:
#               precision    recall  f1-score   support

#            0       0.96      0.98      0.97       640
#            1       0.78      0.67      0.72        69

#     accuracy                           0.95       709
#    macro avg       0.87      0.82      0.85       709
# weighted avg       0.95      0.95      0.95       709

# Confusion Matrix (Validation):
# [[627  13]
#  [ 23  46]]

#  Test Set Results:
#    Accuracy: 0.9478
#    F1 Score (Incompatible Class): 0.7299
#    Confusion Matrix:
# [[622  18]
#  [ 19  50]]

#    Full Classification Report:
#               precision    recall  f1-score   support

#            0       0.97      0.97      0.97       640
#            1       0.74      0.72      0.73        69

#     accuracy                           0.95       709
#    macro avg       0.85      0.85      0.85       709
# weighted avg       0.95      0.95      0.95       709


#  Model saved as '0201_my_rf_model.pkl'