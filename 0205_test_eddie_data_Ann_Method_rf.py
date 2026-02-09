import pandas as pd
import pubchempy as pcp
import cirpy
import joblib
import numpy as np
import sys
import time
from rdkit import Chem
from rdkit import RDLogger
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# 1. Setup & Load Models
RDLogger.DisableLog('rdApp.*')
print("📂 Loading models...")
w2v_model = word2vec.Word2Vec.load('model_300dim.pkl')
rf_model = joblib.load('0201_my_rf_model.pkl')

import pandas as pd

# 直接讀取清洗好的 CSV
df = pd.read_csv('eddie_93_data.csv')

# 取得 CID 列表 
api_cids = df['API_CID'].tolist()
exp_cids = df['Excipient_CID'].tolist()

print(f"Loaded {len(df)} pairs from CSV.")

# 3. Ann's Method: CID -> InChIKey -> SMILES (using cirpy)
# We use a cache to avoid re-fetching duplicates
cid_cache = {}

def get_smiles_ann_method(cid):
    if cid in cid_cache: return cid_cache[cid]
    
    try:
        # Step 1: PubChemPy for InChIKey (Ann's suggestion)
        # Note: Using pcp.Compound.from_cid can still block if run too fast, so we add retry
        c = pcp.Compound.from_cid(int(cid))
        inchikey = c.inchikey
        
        if not inchikey:
            return None
            
        # Step 2: CIRpy for SMILES
        smiles = cirpy.resolve(inchikey, 'smiles')
        
        if smiles:
            cid_cache[cid] = smiles
            return smiles
        else:
            # Fallback: sometimes CIRpy fails but PubChem had it
            return c.isomeric_smiles
            
    except Exception as e:
        print(f"⚠️ Failed for CID {cid}: {e}")
        return None

print("🌍 Converting CIDs to SMILES (Using Ann's cirpy method)...")
tqdm.pandas()
# Collecting all unique CIDs first to handle them efficiently
unique_cids = pd.concat([df['API_CID'], df['Excipient_CID']]).unique()

print(f"   Processing {len(unique_cids)} unique CIDs...")

for cid in tqdm(unique_cids):
    if cid not in cid_cache:
        res = get_smiles_ann_method(cid)
        # Sleep slightly to respect Ann's API method limits
        time.sleep(0.3) 

# Map back
df['API_SMILES'] = df['API_CID'].map(cid_cache)
df['Exp_SMILES'] = df['Excipient_CID'].map(cid_cache)

# Drop failed
df_clean = df.dropna(subset=['API_SMILES', 'Exp_SMILES'])
print(f"✅ Successfully retrieved {len(df_clean)} / {len(df)} pairs.")

# 4. Prediction Logic (Same as before)
print("⚙️ Vectorizing...")
def get_mol2vec(smiles, model):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros(100)
        mol = Chem.AddHs(mol)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        return sentences2vec([sentence], model, unseen='UNK')[0]
    except:
        return np.zeros(100)

if len(df_clean) > 0:
    X_api = np.array([get_mol2vec(s, w2v_model) for s in df_clean['API_SMILES']])
    X_exp = np.array([get_mol2vec(s, w2v_model) for s in df_clean['Exp_SMILES']])
    X_new = np.hstack((X_api, X_exp))
    y_true = df_clean['Label'].values

    print("🔮 Running Predictions...")
    y_pred = rf_model.predict(X_new)

    print("\n" + "="*40)
    print("🧪 Validation Results (Ann's Method)")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
else:
    print("🛑 Still no data. Check internet.")