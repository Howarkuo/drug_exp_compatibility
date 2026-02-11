# import pandas as pd
# import numpy as np
# import joblib
# import sys
# import os
# import time
# import pubchempy as pcp
# import cirpy  # ✅ 新增 CIRpy
# from rdkit import Chem
# from rdkit import RDLogger
# from mordred import Calculator, descriptors
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # 關閉 RDKit 警告
# RDLogger.DisableLog('rdApp.*')

# # ==========================================
# # 1. Configuration
# # ==========================================
# DATA_FILE = '../0210_Compatibility Testset_162.xlsx'
# MODEL_PATH = '0209_mlp_mordred_model.pkl'
# SCALER_PATH = 'mordred_scaler.pkl'
# SELECTOR_PATH = 'mordred_feature_selector.pkl'
# OUTPUT_FILE = '0210_Compatibility Testset_162_results.csv'
# THRESHOLD = 0.25 # ✅ To fix recall, accuracy
# # ==========================================
# # 2. Load Models
# # ==========================================
# print("📂 Loading models...")
# required_files = [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]
# for f in required_files:
#     if not os.path.exists(f):
#         print(f"❌ Missing file: {f}")
#         print("Please run the training script first.")
#         sys.exit(1)

# try:
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     selector = joblib.load(SELECTOR_PATH)
#     print("✅ Models loaded.")
# except Exception as e:
#     print(f"❌ Error loading pickle files: {e}")
#     sys.exit(1)

# # ==========================================
# # 3. Load Data
# # ==========================================
# if not os.path.exists(DATA_FILE):
#     print(f"❌ Data file {DATA_FILE} not found.")
#     sys.exit(1)

# print(f"📂 Reading {DATA_FILE}...")
# df = pd.read_csv(DATA_FILE)

# # 檢查欄位
# required_cols = ['API_CID', 'Excipient_CID', 'Outcome (1: incompatible; 0 compatible)']
# if not all(col in df.columns for col in required_cols):
#     print(f"❌ CSV format error. Must contain: {required_cols}")
#     sys.exit(1)

# print(f"   Loaded {len(df)} rows.")

# # ==========================================
# # 4. Fetch SMILES (Dual Strategy: PubChem + CIRpy)
# # ==========================================
# print("🌍 Fetching SMILES using PubChemPy + CIRpy...")

# def get_smiles_robust(cid):
#     """
#     Robust SMILES fetcher.
#     Strategy 1: PubChemPy direct property (IsomericSMILES)
#     Strategy 2: CID -> InChIKey -> CIRpy (Good for salts like MgO)
#     """
#     cid = int(cid)
#     smi = None
    
#     # Strategy 1: Fast PubChem Property
#     try:
#         props = pcp.get_properties('IsomericSMILES', cid)
#         if props and 'IsomericSMILES' in props[0]:
#             smi = props[0]['IsomericSMILES']
#     except:
#         pass

#     # Strategy 2: If failed or explicitly needed, try InChIKey -> CIRpy
#     if not smi:
#         try:
#             # 1. Get InChIKey from PubChem
#             c = pcp.Compound.from_cid(cid)
#             inchikey = c.inchikey
#             if inchikey:
#                 # 2. Resolve via CIRpy
#                 smi_cir = cirpy.resolve(inchikey, 'smiles')
#                 if smi_cir:
#                     smi = smi_cir
#                     print(f"   🔹 Used CIRpy for CID {cid}: {smi}")
#         except Exception as e:
#             print(f"   ⚠️ Failed to resolve CID {cid}: {e}")
    
#     return smi

# # 為了顯示進度，我們使用 apply
# # 注意：這會比批量抓取慢一點，但對無機物更準確
# from tqdm import tqdm
# tqdm.pandas(desc="Fetching API SMILES")
# df['API_SMILES'] = df['API_CID'].progress_apply(get_smiles_robust)

# tqdm.pandas(desc="Fetching Excipient SMILES")
# df['EXP_SMILES'] = df['Excipient_CID'].progress_apply(get_smiles_robust)

# # 檢查是否有抓不到的
# missing = df[df['API_SMILES'].isnull() | df['EXP_SMILES'].isnull()]
# if not missing.empty:
#     print(f"⚠️ Warning: {len(missing)} rows dropped due to missing SMILES.")
#     df = df.dropna(subset=['API_SMILES', 'EXP_SMILES'])

# if df.empty:
#     print("❌ No valid data left after fetching SMILES.")
#     sys.exit(1)

# print(f"✅ Successfully prepared {len(df)} pairs for prediction.")

# # ==========================================
# # 5. Feature Generation (Mordred)
# # ==========================================
# def calculate_mordred(smiles_list, prefix):
#     print(f"⚙️ Calculating Mordred for {prefix} ({len(smiles_list)} molecules)...")
#     mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
#     # Check for invalid mols
#     valid_mols = []
#     for m in mols:
#         if m: 
#             try:
#                 Chem.SanitizeMol(m)
#                 valid_mols.append(m)
#             except:
#                 valid_mols.append(None)
#         else:
#             valid_mols.append(None)

#     # Calculator
#     calc = Calculator(descriptors, ignore_3D=True)
    
#     # Calculate (n_proc=1 for Windows safety)
#     try:
#         # Filter Nones to avoid crash, then realign
#         clean_mols = [m for m in valid_mols if m is not None]
#         if not clean_mols: return pd.DataFrame()
        
#         df_calc = calc.pandas(clean_mols, nproc=1, quiet=True)
#         df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         # Re-align with original index (handle failed mols)
#         df_full = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_calc.columns)
        
#         # Fill in valid rows
#         valid_indices = [i for i, m in enumerate(valid_mols) if m is not None]
#         df_full.iloc[valid_indices] = df_calc.values
        
#         return df_full.add_prefix(prefix)
        
#     except Exception as e:
#         print(f"❌ Calculation failed: {e}")
#         return pd.DataFrame()

# # 計算特徵
# df_api_feats = calculate_mordred(df['API_SMILES'].tolist(), "API_")
# df_exp_feats = calculate_mordred(df['EXP_SMILES'].tolist(), "EXP_")

# if df_api_feats.empty or df_exp_feats.empty:
#     print("❌ Feature calculation failed.")
#     sys.exit(1)

# # 合併
# X_raw = pd.concat([df_api_feats, df_exp_feats], axis=1)

# # ==========================================
# # 6. Predict & Report
# # ==========================================
# print("🔮 Running Predictions...")

# # 1. Feature Selection
# try:
#     X_selected = selector.transform(X_raw)
# except ValueError as e:
#     print(f"❌ Feature mismatch: {e}")
#     sys.exit(1)

# # 2. Scaling
# X_scaled = scaler.transform(X_selected)

# # 3. Predict
# probs = model.predict_proba(X_scaled)[:, 1]
# preds = (probs >= THRESHOLD).astype(int)

# # 4. Save Results
# df['Probability'] = probs
# df['Predicted_Label'] = preds
# df['Risk_Score'] = [f"{p*100:.1f}%" for p in probs]
# df['Correct'] = df['Label'] == df['Predicted_Label']

# # Report
# y_true = df['Label']
# y_pred = df['Predicted_Label']

# print("\n" + "="*60)
# print(f"🧪 Validation Report (Threshold: {THRESHOLD*100:.0f}%)")
# print("="*60)
# print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
# print("-" * 30)
# print("Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))
# print("-" * 30)
# print("Classification Report:")
# print(classification_report(y_true, y_pred))

# df.to_csv(OUTPUT_FILE, index=False)
# print(f"\n💾 Detailed results saved to: {OUTPUT_FILE}")


# # ============================================================
# # 🧪 Validation Report (Threshold: 25%)
# # ============================================================
# # Accuracy: 0.5604
# # ------------------------------
# # Confusion Matrix:
# # [[40  8]
# #  [32 11]]
# # ------------------------------
# # Classification Report:
# #               precision    recall  f1-score   support

# #            0       0.56      0.83      0.67        48
# #            1       0.58      0.26      0.35        43

# #     accuracy                           0.56        91
# #    macro avg       0.57      0.54      0.51        91
# # weighted avg       0.57      0.56      0.52        91


# # 💾 Detailed results saved to: 0212_eddie_validation_results.csv


import pandas as pd
import numpy as np
import joblib
import sys
import os
import pubchempy as pcp
import cirpy  # ✅ CIRpy for robust salt resolution
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# 關閉 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration
# ==========================================
# 請確保路徑正確
DATA_FILE = '../0210_Compatibility_Testset_162.xlsx' 
MODEL_PATH = '0209_mlp_mordred_model.pkl'
SCALER_PATH = 'mordred_scaler.pkl'
SELECTOR_PATH = 'mordred_feature_selector.pkl'
OUTPUT_FILE = '0210_Compatibility_Testset_162_results.csv'

# 設定嚴格閾值 (Safety First)
THRESHOLD = 0.25 

# ==========================================
# 2. Load Models
# ==========================================
print("📂 Loading models...")
required_files = [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]
for f in required_files:
    if not os.path.exists(f):
        print(f"❌ Missing file: {f}")
        print("Please run the training script (0208) first.")
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
# 3. Load Data (Fixes Applied)
# ==========================================
if not os.path.exists(DATA_FILE):
    print(f"❌ Data file not found: {DATA_FILE}")
    sys.exit(1)

print(f"📂 Reading Excel file: {DATA_FILE}...")

# ⚠️ FIX 1: Use read_excel for .xlsx files
try:
    df = pd.read_excel(DATA_FILE)
except Exception as e:
    print(f"❌ Failed to read Excel file: {e}")
    sys.exit(1)

# ⚠️ FIX 2: Rename Target Column to 'Label'
target_col_name = 'Outcome (1: incompatible; 0 compatible)'
if target_col_name in df.columns:
    df = df.rename(columns={target_col_name: 'Label'})
    print("✅ Column renamed to 'Label'.")
elif 'Label' in df.columns:
    print("✅ 'Label' column found.")
else:
    print(f"❌ Error: Column '{target_col_name}' not found.")
    print(f"   Available columns: {list(df.columns)}")
    sys.exit(1)

# 確保 CID 是整數 (去除空值)
df = df.dropna(subset=['API_CID', 'Excipient_CID'])
df['API_CID'] = df['API_CID'].astype(int)
df['Excipient_CID'] = df['Excipient_CID'].astype(int)

print(f"📊 Loaded {len(df)} pairs for validation.")

# ==========================================
# 4. Fetch SMILES (Dual Strategy)
# ==========================================
print("🌍 Fetching SMILES (PubChem + CIRpy)...")

def get_smiles_robust(cid):
    """
    Robust SMILES fetcher:
    1. Try PubChem IsomericSMILES (Fast)
    2. Try CIRpy (InChIKey -> SMILES) for complex salts
    """
    cid = int(cid)
    smi = None
    
    # Strategy 1: PubChem Property
    try:
        props = pcp.get_properties('IsomericSMILES', cid)
        if props and 'IsomericSMILES' in props[0]:
            smi = props[0]['IsomericSMILES']
    except:
        pass

    # Strategy 2: CIRpy Fallback
    if not smi:
        try:
            c = pcp.Compound.from_cid(cid)
            if c.inchikey:
                smi = cirpy.resolve(c.inchikey, 'smiles')
        except:
            pass
    
    return smi

# Progress Bar
tqdm.pandas()
print("   > Fetching API SMILES...")
df['API_SMILES'] = df['API_CID'].progress_apply(get_smiles_robust)

print("   > Fetching Excipient SMILES...")
df['EXP_SMILES'] = df['Excipient_CID'].progress_apply(get_smiles_robust)

# Remove failed fetches
before_len = len(df)
df = df.dropna(subset=['API_SMILES', 'EXP_SMILES'])
print(f"✅ SMILES Ready: {len(df)} / {before_len} pairs.")

if df.empty:
    print("❌ No valid SMILES found. Exiting.")
    sys.exit(1)

# ==========================================
# 5. Feature Generation (Mordred)
# ==========================================
def calculate_mordred(smiles_list, prefix):
    print(f"⚙️ Calculating Mordred for {prefix}...")
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Filter valid mols
    valid_mols = []
    valid_indices = []
    for i, m in enumerate(mols):
        if m:
            try:
                Chem.SanitizeMol(m)
                valid_mols.append(m)
                valid_indices.append(i)
            except:
                pass
    
    if not valid_mols:
        return pd.DataFrame()

    # Calculate
    calc = Calculator(descriptors, ignore_3D=True)
    df_calc = calc.pandas(valid_mols, nproc=1, quiet=True)
    
    # Fill errors with 0
    df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Align to original length
    df_full = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_calc.columns)
    df_full.iloc[valid_indices] = df_calc.values
    
    return df_full.add_prefix(prefix)

# Generate features
df_api_feats = calculate_mordred(df['API_SMILES'].tolist(), "API_")
df_exp_feats = calculate_mordred(df['EXP_SMILES'].tolist(), "EXP_")

# Concatenate (must match training order: API first, then Excipient)
X_raw = pd.concat([df_api_feats, df_exp_feats], axis=1)

# ==========================================
# 6. Predict & Report
# ==========================================
print("🔮 Predicting...")

# Transform
try:
    X_selected = selector.transform(X_raw)
    X_scaled = scaler.transform(X_selected)
except Exception as e:
    print(f"❌ Feature processing failed: {e}")
    print("Hint: Ensure Mordred versions match and no features are missing.")
    sys.exit(1)

# Predict
probs = model.predict_proba(X_scaled)[:, 1]
preds = (probs >= THRESHOLD).astype(int)

# Save Results
df['Probability'] = probs
df['Predicted_Label'] = preds
df['Risk_Score'] = [f"{p*100:.1f}%" for p in probs]
df['Is_Correct'] = (df['Label'] == df['Predicted_Label'])

# Metrics
y_true = df['Label']
y_pred = df['Predicted_Label']

print("\n" + "="*60)
print(f"🧪 Final Report (Threshold: {THRESHOLD:.2f})")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print("-" * 30)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("-" * 30)
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Export
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Results saved to: {OUTPUT_FILE}")