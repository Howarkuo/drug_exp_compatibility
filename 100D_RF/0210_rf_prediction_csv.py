import pandas as pd
import numpy as np
import joblib
import sys
import os
import pubchempy as pcp
import cirpy  # ✅ CIRpy for robust salt resolution
from rdkit import Chem
from rdkit import RDLogger
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# 關閉 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration
# ==========================================
RAW_DATA_FILE = '../0210_Compatibility_Testset_162.xlsx'  # 原始 Excel
OUTPUT_FILE = '0210_Compatibility_Testset_162_mol2vec_results.csv'

RF_MODEL_PATH = '0201_my_rf_model.pkl'
W2V_MODEL_PATH = 'model_300dim.pkl'

# ==========================================
# 2. Load Models
# ==========================================
print("📂 Loading RF + Mol2Vec models...")

if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(W2V_MODEL_PATH):
    print(f"❌ Missing model files: {RF_MODEL_PATH} or {W2V_MODEL_PATH}")
    sys.exit(1)

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    w2v_model = word2vec.Word2Vec.load(W2V_MODEL_PATH)
    print("✅ Models loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

# ==========================================
# 3. Load Raw Data
# ==========================================
print(f"📂 Reading raw Excel: {RAW_DATA_FILE}...")
if not os.path.exists(RAW_DATA_FILE):
    print(f"❌ File not found: {RAW_DATA_FILE}")
    sys.exit(1)

try:
    df = pd.read_excel(RAW_DATA_FILE)
    
    # 標準化 Label 欄位
    target_col = 'Outcome (1: incompatible; 0 compatible)'
    if target_col in df.columns:
        df = df.rename(columns={target_col: 'Label'})
    
    # 確保 CID 是整數
    df = df.dropna(subset=['API_CID', 'Excipient_CID'])
    df['API_CID'] = df['API_CID'].astype(int)
    df['Excipient_CID'] = df['Excipient_CID'].astype(int)

    print(f"📊 Loaded {len(df)} pairs.")
except Exception as e:
    print(f"❌ Failed to read Excel: {e}")
    sys.exit(1)

# ==========================================
# 4. Fetch SMILES (Live Fetching)
# ==========================================
print("🌍 Fetching SMILES from PubChem + CIRpy (this may take 1-2 mins)...")

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

    # Strategy 2: CIRpy Fallback (Good for Mg salts)
    if not smi:
        try:
            c = pcp.Compound.from_cid(cid)
            if c.inchikey:
                smi = cirpy.resolve(c.inchikey, 'smiles')
        except:
            pass
    
    return smi

# 使用 tqdm 顯示進度條
tqdm.pandas(desc="Fetching API SMILES")
df['Target_API_SMILES'] = df['API_CID'].progress_apply(get_smiles_robust)

tqdm.pandas(desc="Fetching Excipient SMILES")
df['Target_EXP_SMILES'] = df['Excipient_CID'].progress_apply(get_smiles_robust)

# 清除抓不到的資料
before_len = len(df)
df = df.dropna(subset=['Target_API_SMILES', 'Target_EXP_SMILES'])
print(f"✅ SMILES Ready: {len(df)} / {before_len} pairs.")

if df.empty:
    print("❌ No valid SMILES found. Exiting.")
    sys.exit(1)

# ==========================================
# 5. Feature Engineering (Mol2Vec)
# ==========================================
def get_mol2vec_vector(smiles, model):
    """Convert SMILES to 300-dim vector"""
    try:
        mol = Chem.MolFromSmiles(str(smiles)) 
        if not mol: return None
        mol = Chem.AddHs(mol)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        vector = sentences2vec([sentence], model, unseen='UNK')[0]
        return vector
    except:
        return None

print("⚙️ Vectorizing molecules (Mol2Vec)...")
X_data = []
valid_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Vectorizing"):
    api_vec = get_mol2vec_vector(row['Target_API_SMILES'], w2v_model)
    exp_vec = get_mol2vec_vector(row['Target_EXP_SMILES'], w2v_model)
    
    if api_vec is not None and exp_vec is not None:
        combined_vec = np.concatenate([api_vec, exp_vec])
        X_data.append(combined_vec)
        valid_indices.append(idx)

# Filter DF
df_clean = df.loc[valid_indices].copy()
X = np.array(X_data)

if len(X) == 0:
    print("❌ No valid vectors generated.")
    sys.exit(1)

# ==========================================
# 6. Prediction & Evaluation
# ==========================================
print("🔮 Predicting with Random Forest...")

if 'Label' in df_clean.columns:
    y_true = df_clean['Label'].astype(int)
    has_label = True
else:
    print("⚠️ Warning: No ground truth label found. Metrics will be skipped.")
    has_label = False

y_pred = rf_model.predict(X)
y_probs = rf_model.predict_proba(X)[:, 1]

# Save results
df_clean['RF_Prob'] = y_probs
df_clean['RF_Pred'] = y_pred

if has_label:
    df_clean['RF_Correct'] = (y_true == y_pred)
    
    print("\n" + "="*60)
    print("🧪 Benchmark Report: RF + Mol2Vec")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Calculate Critical Recall (Class 1)
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
        recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n🔥 Critical Recall (Class 1 - Incompatible): {recall_class1:.4f}")

# Save
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Results saved to: {OUTPUT_FILE}")