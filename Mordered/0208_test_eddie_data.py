import pandas as pd
import numpy as np
import joblib
import sys
import os
import time
import pubchempy as pcp
import cirpy  # ✅ 新增 CIRpy
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 關閉 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration
# ==========================================
DATA_FILE = '../eddie_100_data.csv'
MODEL_PATH = '0209_mlp_mordred_model.pkl'
SCALER_PATH = 'mordred_scaler.pkl'
SELECTOR_PATH = 'mordred_feature_selector.pkl'
OUTPUT_FILE = '0212_eddie_validation_results.csv'
THRESHOLD = 0.25 # ✅ To fix recall, accuracy
# ==========================================
# 2. Load Models
# ==========================================
print("📂 Loading models...")
required_files = [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]
for f in required_files:
    if not os.path.exists(f):
        print(f"❌ Missing file: {f}")
        print("Please run the training script first.")
        sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    print("✅ Models loaded.")
except Exception as e:
    print(f"❌ Error loading pickle files: {e}")
    sys.exit(1)

# ==========================================
# 3. Load Data
# ==========================================
if not os.path.exists(DATA_FILE):
    print(f"❌ Data file {DATA_FILE} not found.")
    sys.exit(1)

print(f"📂 Reading {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# 檢查欄位
required_cols = ['API_CID', 'Excipient_CID', 'Label']
if not all(col in df.columns for col in required_cols):
    print(f"❌ CSV format error. Must contain: {required_cols}")
    sys.exit(1)

print(f"   Loaded {len(df)} rows.")

# ==========================================
# 4. Fetch SMILES (Dual Strategy: PubChem + CIRpy)
# ==========================================
print("🌍 Fetching SMILES using PubChemPy + CIRpy...")

def get_smiles_robust(cid):
    """
    Robust SMILES fetcher.
    Strategy 1: PubChemPy direct property (IsomericSMILES)
    Strategy 2: CID -> InChIKey -> CIRpy (Good for salts like MgO)
    """
    cid = int(cid)
    smi = None
    
    # Strategy 1: Fast PubChem Property
    try:
        props = pcp.get_properties('IsomericSMILES', cid)
        if props and 'IsomericSMILES' in props[0]:
            smi = props[0]['IsomericSMILES']
    except:
        pass

    # Strategy 2: If failed or explicitly needed, try InChIKey -> CIRpy
    if not smi:
        try:
            # 1. Get InChIKey from PubChem
            c = pcp.Compound.from_cid(cid)
            inchikey = c.inchikey
            if inchikey:
                # 2. Resolve via CIRpy
                smi_cir = cirpy.resolve(inchikey, 'smiles')
                if smi_cir:
                    smi = smi_cir
                    print(f"   🔹 Used CIRpy for CID {cid}: {smi}")
        except Exception as e:
            print(f"   ⚠️ Failed to resolve CID {cid}: {e}")
    
    return smi

# 為了顯示進度，我們使用 apply
# 注意：這會比批量抓取慢一點，但對無機物更準確
from tqdm import tqdm
tqdm.pandas(desc="Fetching API SMILES")
df['API_SMILES'] = df['API_CID'].progress_apply(get_smiles_robust)

tqdm.pandas(desc="Fetching Excipient SMILES")
df['EXP_SMILES'] = df['Excipient_CID'].progress_apply(get_smiles_robust)

# 檢查是否有抓不到的
missing = df[df['API_SMILES'].isnull() | df['EXP_SMILES'].isnull()]
if not missing.empty:
    print(f"⚠️ Warning: {len(missing)} rows dropped due to missing SMILES.")
    df = df.dropna(subset=['API_SMILES', 'EXP_SMILES'])

if df.empty:
    print("❌ No valid data left after fetching SMILES.")
    sys.exit(1)

print(f"✅ Successfully prepared {len(df)} pairs for prediction.")

# ==========================================
# 5. Feature Generation (Mordred)
# ==========================================
def calculate_mordred(smiles_list, prefix):
    print(f"⚙️ Calculating Mordred for {prefix} ({len(smiles_list)} molecules)...")
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Check for invalid mols
    valid_mols = []
    for m in mols:
        if m: 
            try:
                Chem.SanitizeMol(m)
                valid_mols.append(m)
            except:
                valid_mols.append(None)
        else:
            valid_mols.append(None)

    # Calculator
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Calculate (n_proc=1 for Windows safety)
    try:
        # Filter Nones to avoid crash, then realign
        clean_mols = [m for m in valid_mols if m is not None]
        if not clean_mols: return pd.DataFrame()
        
        df_calc = calc.pandas(clean_mols, nproc=1, quiet=True)
        df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Re-align with original index (handle failed mols)
        df_full = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_calc.columns)
        
        # Fill in valid rows
        valid_indices = [i for i, m in enumerate(valid_mols) if m is not None]
        df_full.iloc[valid_indices] = df_calc.values
        
        return df_full.add_prefix(prefix)
        
    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return pd.DataFrame()

# 計算特徵
df_api_feats = calculate_mordred(df['API_SMILES'].tolist(), "API_")
df_exp_feats = calculate_mordred(df['EXP_SMILES'].tolist(), "EXP_")

if df_api_feats.empty or df_exp_feats.empty:
    print("❌ Feature calculation failed.")
    sys.exit(1)

# 合併
X_raw = pd.concat([df_api_feats, df_exp_feats], axis=1)

# ==========================================
# 6. Predict & Report
# ==========================================
print("🔮 Running Predictions...")

# 1. Feature Selection
try:
    X_selected = selector.transform(X_raw)
except ValueError as e:
    print(f"❌ Feature mismatch: {e}")
    sys.exit(1)

# 2. Scaling
X_scaled = scaler.transform(X_selected)

# 3. Predict
probs = model.predict_proba(X_scaled)[:, 1]
preds = (probs >= THRESHOLD).astype(int)

# 4. Save Results
df['Probability'] = probs
df['Predicted_Label'] = preds
df['Risk_Score'] = [f"{p*100:.1f}%" for p in probs]
df['Correct'] = df['Label'] == df['Predicted_Label']

# Report
y_true = df['Label']
y_pred = df['Predicted_Label']

print("\n" + "="*60)
print(f"🧪 Validation Report (Threshold: {THRESHOLD*100:.0f}%)")
print("="*60)
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("-" * 30)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("-" * 30)
print("Classification Report:")
print(classification_report(y_true, y_pred))

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Detailed results saved to: {OUTPUT_FILE}")


# ============================================================
# 🧪 Validation Report (Threshold: 25%)
# ============================================================
# Accuracy: 0.5604
# ------------------------------
# Confusion Matrix:
# [[40  8]
#  [32 11]]
# ------------------------------
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.56      0.83      0.67        48
#            1       0.58      0.26      0.35        43

#     accuracy                           0.56        91
#    macro avg       0.57      0.54      0.51        91
# weighted avg       0.57      0.56      0.52        91


# 💾 Detailed results saved to: 0212_eddie_validation_results.csv