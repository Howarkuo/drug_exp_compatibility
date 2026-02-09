import pandas as pd
import requests
import joblib
import numpy as np
import sys
import time
from rdkit import Chem
from rdkit import RDLogger
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Setup
RDLogger.DisableLog('rdApp.*')
print("📂 Loading models...")
try:
    w2v_model = word2vec.Word2Vec.load('model_300dim.pkl')
    rf_model = joblib.load('0201_my_rf_model.pkl')
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

# 2. Raw Data Parsing
raw_data = """
54678486
3084116
1
1775
14792
1
5560
14792
1
5324346
6134
1
1986
6134
1
1986
6251
1
657302
6134
1
657302
24497
1
5280343
11029
1
1986
444041
1
1986
90265172
1
1986
8567
1
1986
11177
0
1986
51063134
0
1986
154932
0
969516
107676
0
969516
24751
0
969516
444041
0
638024
107676
0
638024
24751
0
638024
444041
0
5280343
107676
0
5280343
24751
0
5280343
444041
0
31553
107676
0
31553
24751
0
31553
444041
0
3715
11147
0
489181
169446502
0
489181
10850
0
489181
5280450
0
489181
17472
0
5743
57503849
0
5743
6328154
0
5743
1030
0
5311309
6134
0
5311309
23678829
0
5311309
6917
0
5311309
26924
0
5311309
11177
0
135398513
11177
1
135398513
5793
1
135398513
5988
1
135398513
47207535
1
135398513
6328154
1
135398513
23665634
0
33741
23665634
0
4485
23665634
0
4485
6917
1
4485
14055602
0
3883
11177
1
3883
6917
1
3883
10129990
1
3883
24083
1
3883
5360315
1
3883
14792
0
4044
11177
1
4044
6251
0
135413523
14055602
0
135413523
6328154
0
135413523
8456
1
135413523
54670067
1
135413523
311
1
656846
6134
0
656846
14055602
0
656846
11177
0
656846
6251
1
656846
47207535
0
656846
3423265
0
18283
311
1
14985
123289
1
14985
11029
1
14985
10340
1
14985
6917
1
14985
14055602
0
14985
54680660
0
14985
26042
0
14985
24441
0
20058
11177
0
20058
14055602
0
5702160
26042
1
54670067
11177
1
54670067
47207535
1
54670067
5281
0
54670067
6251
0
54670067
23668193
1
4101
54670067
1
5284443
54670067
1
5284451
54670067
1
16760658
54670067
1
4814
54670067
1
5281
73981
1
5281
6093208
1
"""

lines = [x.strip() for x in raw_data.strip().split('\n') if x.strip()]
data_list = []
for i in range(0, len(lines), 3):
    if i+2 < len(lines):
        data_list.append({
            'API_CID': int(lines[i]),
            'Excipient_CID': int(lines[i+1]),
            'Label': int(lines[i+2])
        })

df_new = pd.DataFrame(data_list)
print(f"📊 Parsed {len(df_new)} rows of data.")

# 3. Robust Batch Fetching Function
def get_smiles_batch(cid_list):
    """
    Fetches SMILES for a list of CIDs using direct PubChem API.
    Handles chunks to avoid URL length limits.
    """
    unique_cids = list(set(cid_list))
    smiles_map = {}
    chunk_size = 50 # Safe chunk size for API
    
    print(f"🌍 Fetching {len(unique_cids)} CIDs via Direct API...")

    for i in range(0, len(unique_cids), chunk_size):
        chunk = unique_cids[i:i + chunk_size]
        cid_str = ",".join(map(str, chunk))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IsomericSMILES,CanonicalSMILES/JSON"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for prop in data['PropertyTable']['Properties']:
                    cid = prop['CID']
                    # Try Isomeric first, fallback to Canonical
                    smi = prop.get('IsomericSMILES', prop.get('CanonicalSMILES'))
                    if smi:
                        smiles_map[cid] = smi
            else:
                print(f"⚠️ API Error {r.status_code} for chunk {i}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            time.sleep(1) # Backoff
            
        time.sleep(0.5) # Politeness delay
        
    return smiles_map

# Execute Fetching
all_cids = df_new['API_CID'].tolist() + df_new['Excipient_CID'].tolist()
smiles_dict = get_smiles_batch(all_cids)

# Map back
df_new['API_SMILES'] = df_new['API_CID'].map(smiles_dict)
df_new['Exp_SMILES'] = df_new['Excipient_CID'].map(smiles_dict)

# Check results
df_clean = df_new.dropna(subset=['API_SMILES', 'Exp_SMILES'])
print(f"✅ Successfully retrieved {len(df_clean)} / {len(df_new)} pairs.")

if len(df_clean) == 0:
    print("🛑 CRITICAL: Still cannot fetch SMILES. Check your internet/firewall.")
    sys.exit(1)

# 4. Feature Engineering
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

X_api = np.array([get_mol2vec(s, w2v_model) for s in df_clean['API_SMILES']])
X_exp = np.array([get_mol2vec(s, w2v_model) for s in df_clean['Exp_SMILES']])
X_new = np.hstack((X_api, X_exp))
y_true = df_clean['Label'].values

# 5. Predict
# the predict() in scikit-learn
# this is where the 50 % threshold hardcoded 
print("🔮 Running Predictions...")
y_pred = rf_model.predict(X_new)

# 6. Results
print("\n" + "="*40)
print("🧪 External Validation Results (Eddie's Data)")
print("="*40)
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Save results
df_clean['Predicted'] = y_pred
df_clean.to_csv('0205_eddie_validation_results.csv', index=False)