# import pandas as pd
# import numpy as np
# import joblib
# import sys
# from rdkit import Chem
# from mordred import Calculator, descriptors

# # ==========================================
# # 1. Configuration
# # ==========================================
# # MODEL_PATH = '0208_modelered_rf_model.pkl'  # Your new Mordred Model
# MODEL_PATH = '0209_mlp_mordred_model.pkl'
# SCALER_PATH = 'mordred_scaler.pkl'
# SELECTOR_PATH = 'mordred_feature_selector.pkl'
# # Define your inputs
# API_NAME = "Vitamin C (Ascorbic Acid)"
# API_SMILES = "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O"

# EXCIPIENTS_LIST = [
#     ("Mg Stearate", "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"),
#     ("Fluorinated Amide", "CCCS(=O)(=O)CC(=O)NC1=CC=C(C=C1)F"),
#     ("Cellulose", "COC1C(OC(C(C1O)O)OC2C(OC(C(C2O)O)OC)CO)CO"),
#     ("Stearic Acid", "CCCCCCCCCCCCCCCCCC(=O)O"),
#     ("Mannitol", "C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O"),
#     ("Silicon Dioxide", "O=[Si]=O")
# ]

# # ==========================================
# # 2. Helper Function (Calculates Mordred Features)
# # ==========================================
# def get_mordred_features(smiles, prefix):
#     # Setup Calculator (Same as training)
#     calc = Calculator(descriptors, ignore_3D=True)
    
#     # Convert to Mol
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return None
    
#     # Calculate 1613 features
#     # nproc=1 is safer for small batches/Windows
#     df = calc.pandas([mol], nproc=1, quiet=True)
    
#     # Clean data (same as training)
#     df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
#     # Rename columns to match training format (API_..., EXP_...)
#     df = df.add_prefix(prefix)
#     return df

# # ==========================================
# # 3. Main Prediction Logic
# # ==========================================
# print(f"📂 Loading model: {MODEL_PATH}...")
# try:
#     rf_model = joblib.load(MODEL_PATH)
#     print("   ✅ Model loaded successfully.")
# except FileNotFoundError:
#     print("   ❌ Model not found. Check the filename.")
#     sys.exit()

# # --- CRITICAL STEP: GET TRAINED FEATURE NAMES ---
# # The model knows which 2738 columns it needs.
# try:
#     required_columns = rf_model.feature_names_in_
#     print(f"   ℹ️  Model expects {len(required_columns)} features.")
# except AttributeError:
#     print("   ❌ Error: This model doesn't have feature names saved.")
#     print("   Did you train it using a DataFrame? (The code I gave you does this).")
#     sys.exit()

# print("\n" + "="*70)
# print(f"🧪 COMPATIBILITY PREDICTION (Mordred Model)")
# print(f"💊 API: {API_NAME}")
# print("="*70)
# print(f"{'Excipient':<20} | {'Prediction':<15} | {'Probability':<10}")
# print("-" * 70)

# # 1. Calculate API Features (Once)
# df_api = get_mordred_features(API_SMILES, "API_")

# if df_api is None:
#     print("❌ Invalid API SMILES")
#     sys.exit()

# # 2. Loop Excipients
# for exp_name, exp_smiles in EXCIPIENTS_LIST:
    
#     # Calculate Excipient Features
#     df_exp = get_mordred_features(exp_smiles, "EXP_")
    
#     if df_exp is None:
#         print(f"{exp_name:<20} | ❌ Invalid SMILES")
#         continue

#     # 3. Combine API + Excipient
#     # We reset index to allow horizontal concat
#     df_combined = pd.concat([df_api.reset_index(drop=True), df_exp.reset_index(drop=True)], axis=1)

#     # 4. ALIGN COLUMNS (The Magic Step)
#     # The raw calculation gives ~3200 columns, but the model only wants the 2738 non-zero ones.
#     # We reindex to keep only the required columns, filling missing ones with 0.
#     df_final = df_combined.reindex(columns=required_columns, fill_value=0)
    
#     # 5. Predict
#     prediction_class = rf_model.predict(df_final)[0]
#     probabilities = rf_model.predict_proba(df_final)[0]
    
#     # Class 0 = Compatible, 1 = Incompatible (Check your label mapping!)
#     # Usually: 0=Compatible, 1=Incompatible
#     confidence = probabilities[prediction_class] * 100
    
#     if prediction_class == 1:
#         result_str = "🔴 Incompatible"
#     else:
#         result_str = "🟢 Compatible"

#     print(f"{exp_name:<20} | {result_str:<15} | {confidence:.2f}%")

# print("-" * 70)


# #Correct Answer:
# # Mg -Stearate: Incompatible
# # Fluorinated Amide : Incompatible
# # Cellulose: Compatible
# # Stearic Acid: Compatible
# # Mannitol" Compatible
# # Silicon Dioxide: Compatible

import pandas as pd
import numpy as np
import joblib
import sys
import os
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors

# 關閉 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration
# ==========================================
# 請確保這些檔名與您訓練產出的檔案一致
MODEL_PATH = '0209_mlp_mordred_model.pkl'
SCALER_PATH = 'mordred_scaler.pkl'
SELECTOR_PATH = 'mordred_feature_selector.pkl'

# 定義測試案例 (Vitamin C vs Others)
API_NAME = "Vitamin C"
API_SMILES = "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O"

test_cases = [
    ("Mg Stearate", "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]", "Incompatible"),
    ("Fluorinated Amide", "CCCS(=O)(=O)CC(=O)NC1=CC=C(C=C1)F", "Incompatible"),
    ("Cellulose", "OC1C(OC(C(C1O)O)OC2C(OC(C(C2O)O)OC)CO)CO", "Compatible"), 
    ("Stearic Acid", "CCCCCCCCCCCCCCCCCC(=O)O", "Compatible"),
    ("Mannitol", "C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O", "Compatible"),
    ("Silicon Dioxide", "O=[Si]=O", "Compatible")
]

# ==========================================
# 2. Load Models
# ==========================================
print("📂 Loading models...")
required_files = [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]

for f in required_files:
    if not os.path.exists(f):
        print(f"❌ Missing file: {f}")
        print("Please run the training script '0208_train_MLP.py' first.")
        sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading pickle files: {e}")
    sys.exit(1)

# ==========================================
# 3. Feature Generation (Mordred)
# ==========================================
def get_mordred_features(smiles_list):
    # 1. Convert to Mol
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # 2. Calculator (Must match training config exactly)
    calc = Calculator(descriptors, ignore_3D=True)
    
    try:
        # Use n_proc=1 for Windows safety
        df = calc.pandas(mols, nproc=1, quiet=True)
        
        # 3. Clean Data (Force numeric)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        print(f"❌ Error in Mordred calculation: {e}")
        return pd.DataFrame()

print(f"⚙️ Calculating descriptors for {len(test_cases)} pairs...")

# 產生 API 特徵
api_df = get_mordred_features([API_SMILES])
# 產生 Excipient 特徵
exp_df = get_mordred_features([case[1] for case in test_cases])

if api_df.empty or exp_df.empty:
    print("❌ Feature generation failed.")
    sys.exit(1)

# 擴展 API 特徵以匹配 Excipient 數量
api_df_expanded = pd.concat([api_df]*len(exp_df), ignore_index=True)

# 合併 (API + Excipient) -> 這是 X_raw
# 注意：這裡的欄位順序必須與訓練時完全一樣 (Mordred 保證了這點)
api_df_expanded = api_df_expanded.add_prefix('API_')
exp_df = exp_df.add_prefix('EXP_')
X_raw = pd.concat([api_df_expanded, exp_df], axis=1)

# ==========================================
# 4. Processing Pipeline
# ==========================================
print("🔄 Processing features...")

# 1. Feature Selection (VarianceThreshold)
# 這一步會自動過濾掉訓練時被刪除的欄位
try:
    X_selected = selector.transform(X_raw)
except ValueError as e:
    print(f"❌ Dimension mismatch: {e}")
    print(f"   Input shape: {X_raw.shape}")
    print(f"   Selector expected: {selector.n_features_in_} features")
    sys.exit(1)

# 2. Scaling (StandardScaler)
X_scaled = scaler.transform(X_selected)

# 3. Predict (MLP)
print("🔮 Predicting...")
probs = model.predict_proba(X_scaled)[:, 1]

# 設定 30% 為警戒閾值 (因為這攸關安全)
THRESHOLD = 0.5
preds = (probs >= THRESHOLD).astype(int)

# ==========================================
# 5. Final Report
# ==========================================
print("\n" + "="*80)
print(f"🧪 MLP + Mordred Compatibility Report (Threshold: {THRESHOLD*100:.0f}%)")
print(f"💊 API: {API_NAME}")
print("="*80)
print(f"{'Excipient':<20} | {'True Label':<12} | {'Prediction':<12} | {'Risk Score':<5} | {'Result'}")
print("-" * 80)

for i, (name, smi, true_label) in enumerate(test_cases):
    p = probs[i]
    pred_label = "Incompatible" if preds[i] == 1 else "Compatible"
    
    # Check correctness
    is_correct = (pred_label == true_label)
    icon = "✅" if is_correct else "❌"
    
    # Risk Bar Visualization
    bar_len = int(p * 10)
    bar = "█" * bar_len + "░" * (10 - bar_len)
    
    # Color code risk score (Text based)
    risk_str = f"{p*100:.1f}%"
    
    print(f"{name:<20} | {true_label:<12} | {pred_label:<12} | {risk_str:<6} {bar} | {icon}")

print("="*80)
print("✅ Done.")




# ================================================================================
# 🧪 MLP + Mordred Compatibility Report (Threshold: 50%)
# 💊 API: Vitamin C
# ================================================================================
# Excipient            | True Label   | Prediction   | Risk Score | Result
# --------------------------------------------------------------------------------
# Mg Stearate          | Incompatible | Compatible   | 20.6%  ██░░░░░░░░ | ❌
# Fluorinated Amide    | Incompatible | Incompatible | 51.0%  █████░░░░░ | ✅
# Cellulose            | Compatible   | Compatible   | 1.3%   ░░░░░░░░░░ | ✅
# Stearic Acid         | Compatible   | Compatible   | 0.3%   ░░░░░░░░░░ | ✅
# Mannitol             | Compatible   | Compatible   | 0.2%   ░░░░░░░░░░ | ✅
# Silicon Dioxide      | Compatible   | Compatible   | 2.1%   ░░░░░░░░░░ | ✅
# ================================================================================