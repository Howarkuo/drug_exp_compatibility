import pandas as pd
import numpy as np
import joblib
import os
import sys
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
DATA_PATH = '../2024_Drug_compatibility_dataset.xlsx'
CACHE_PATH = '0208_train_data_mordred_features.csv'

if not os.path.exists(DATA_PATH):
    print(f"❌ Cannot find {DATA_PATH}")
    sys.exit(1)

# Load Data
try:
    if DATA_PATH.endswith('.xlsx'):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded dataset: {len(df)} rows")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    sys.exit(1)

# ==========================================
# 2. Mordred Feature Generation (Pure Python)
# ==========================================
def generate_mordred_features(smiles_list, prefix):
    """
    Generate 2D descriptors using Mordred (No Java required).
    """
    print(f"⚙️ Generating Mordred descriptors for {prefix}...")
    
    # 1. Convert SMILES to RDKit Mols
    mols = []
    valid_indices = []
    for i, s in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(s)
            # Basic sanitization
            if mol:
                Chem.SanitizeMol(mol)
                mols.append(mol)
                valid_indices.append(i)
            else:
                mols.append(None)
        except:
            mols.append(None)

    # 2. Setup Calculator (2D only, ignore 3D to save time)
    calc = Calculator(descriptors, ignore_3D=True)
    
    # 3. Calculate (returns pandas DataFrame directly)
    # n_proc=-1 means use all CPUs
    try:
        # Filter out Nones for calculation
        valid_mols = [m for m in mols if m is not None]
        
        if not valid_mols:
            print(f"⚠️ No valid molecules for {prefix}")
            return pd.DataFrame()

        df_calc = calc.pandas(valid_mols, nproc=1, quiet=True)
        
        # 4. Handle Errors & Non-numeric values
        # Mordred puts error objects in cells where calculation failed
        # We coerce them to NaN, then fill with 0
        df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 5. Re-align with original list (handle invalid SMILES)
        # Create a full-length dataframe filled with 0s
        df_full = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_calc.columns)
        
        # Update rows that were successfully calculated
        df_full.iloc[valid_indices] = df_calc.values
        
        # Add prefix
        df_full = df_full.add_prefix(prefix)
        
        print(f"   ✅ Generated {df_full.shape[1]} features for {prefix}")
        return df_full

    except Exception as e:
        print(f"❌ Mordred Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. Feature Engineering
# ==========================================
if os.path.exists(CACHE_PATH):
    print(f"✅ Found cached features at {CACHE_PATH}, loading...")
    df_final = pd.read_csv(CACHE_PATH)
else:
    print("🚀 No cache found. Starting Mordred calculation...")
    
    # Calculate Features
    df_api = generate_mordred_features(df['API_Smiles'].tolist(), prefix='API_')
    df_exp = generate_mordred_features(df['Excipient_Smiles'].tolist(), prefix='EXP_')
    
    if df_api.empty or df_exp.empty:
        print("🛑 Critical Error: Feature generation failed.")
        sys.exit(1)

    # Merge
    df_features = pd.concat([df_api, df_exp], axis=1)
    
    # Add Label
    target_col = 'Outcome (1: incompatible; 0 compatible)'
    if target_col not in df.columns:
        cols = [c for c in df.columns if 'outcome' in c.lower() or 'label' in c.lower()]
        if cols: target_col = cols[0]
            
    df_features['Label'] = df[target_col]
    
    # Save cache
    df_features.to_csv(CACHE_PATH, index=False)
    print(f"💾 Features saved to {CACHE_PATH}")
    df_final = df_features

# ==========================================
# 4. Processing & Training (Same as before)
# ==========================================
X = df_final.drop(columns=['Label'])
y = df_final['Label']

# Remove constant features
print("Sweep: Removing constant features...")
selector = VarianceThreshold(threshold=0)
X_reduced = selector.fit_transform(X)
print(f"   Features reduced from {X.shape[1]} to {X_reduced.shape[1]}")

joblib.dump(selector, 'mordred_feature_selector.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'mordred_scaler.pkl')

# SMOTE
print(f"⚖️ Applying SVM-SMOTE...")
try:
    smote = SVMSMOTE(random_state=42, k_neighbors=5, m_neighbors=10, out_step=0.5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
except:
    print("⚠️ SVM-SMOTE failed, falling back to standard SMOTE")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# MLP Training
print("🔥 Training MLP Classifier...")
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)], 
    'solver': ['adam', 'lbfgs'],
    'activation': ['relu'],
    'alpha': [0.0001, 0.01],
    'max_iter': [500] # Increase this to give it time to learn
}

mlp = MLPClassifier(random_state=42, early_stopping=True)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='f1', n_jobs=-1)

grid_search.fit(X_train_bal, y_train_bal)
best_mlp = grid_search.best_estimator_
print(f"🏆 Best Parameters: {grid_search.best_params_}")

# Evaluation
print("\n" + "="*40)
print("🧪 Test Set Evaluation (MLP + Mordred)")
print("="*40)
y_pred = best_mlp.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(best_mlp, '0209_mlp_mordred_model.pkl')
print("💾 Model saved.")


# 0208_train_data_mordred_features.csv

# Sweep: Removing constant features...

#    Features reduced from 3226 to 2738

# ⚖️ Scaling features...

# ⚖️ Applying SVM-SMOTE...

# 🔥 Training MLP Classifier...

# 🏆 Best Parameters: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'solver': 'adam'}



# ========================================

# 🧪 Test Set Evaluation (MLP + Mordred)

# ========================================

# Accuracy: 0.9478



# Confusion Matrix:

# [[622  18]

#  [ 19  50]]



# Classification Report:

#               precision    recall  f1-score   support



#            0       0.97      0.97      0.97       640

#            1       0.74      0.72      0.73        69



#     accuracy                           0.95       709

#    macro avg       0.85      0.85      0.85       709

# weighted avg       0.95      0.95      0.95       709