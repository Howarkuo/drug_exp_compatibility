import pandas as pd
import numpy as np
import joblib
import sys
from rdkit import Chem
from rdkit import RDLogger
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

RF_MODEL_PATH = '0201_my_rf_model.pkl'
W2V_MODEL_PATH = 'model_300dim.pkl'

# Define your inputs (API: Vitamin C)
API_NAME = "Vitamin C (Ascorbic Acid)"
API_SMILES = "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O"

# List of Excipients to test
# Format: (Name, SMILES)
EXCIPIENTS_LIST = [
    ("Mg Stearate", "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"),
    ("Fluorinated Amide", "CCCS(=O)(=O)CC(=O)NC1=CC=C(C=C1)F"),
    ("Cellulose", "COC1C(OC(C(C1O)O)OC2C(OC(C(C2O)O)OC)CO)CO"),
    ("Stearic Acid", "CCCCCCCCCCCCCCCCCC(=O)O"),
    ("Mannitol", "C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O"),
    ("Silicon Dioxide", "O=[Si]=O")
]

# ==========================================
# 2. Load Models
# ==========================================
print("📂 Loading models...")

# Load Random Forest
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print(f"   ✅ Loaded RF Model: {RF_MODEL_PATH}")
except FileNotFoundError:
    print(f"   ❌ Error: {RF_MODEL_PATH} not found. Please check filename.")
    sys.exit(1)

# Load Word2Vec (Needed to convert SMILES to Vectors)
try:
    w2v_model = word2vec.Word2Vec.load(W2V_MODEL_PATH)
    print(f"   ✅ Loaded Word2Vec: {W2V_MODEL_PATH}")
except FileNotFoundError:
    print(f"   ❌ Error: {W2V_MODEL_PATH} not found.")
    sys.exit(1)

# ==========================================
# 3. Feature Engineering Helper
# ==========================================
def get_mol2vec_vector(smiles, model):
    """
    Converts a SMILES string into a 300-dim vector using Mol2Vec.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # Standardize molecule (Add Hydrogens is crucial for Mol2Vec)
        mol = Chem.AddHs(mol)
        
        # Convert to "Sentence" (Mol2Vec format)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        
        # Convert to Vector (Returns a matrix, we take the first row)
        # unseen='UNK' handles atoms/substructures not seen during training
        vector = sentences2vec([sentence], model, unseen='UNK')[0]
        return vector
        
    except Exception as e:
        print(f"   ⚠️ Error vectorizing {smiles}: {e}")
        return None

# ==========================================
# 4. Run Prediction Loop
# ==========================================
print("\n" + "="*70)
print(f"🧪 COMPATIBILITY PREDICTION REPORT")
print(f"💊 API: {API_NAME}")
print("="*70)
print(f"{'Excipient':<20} | {'Prediction':<15} | {'Probability':<10}")
print("-" * 70)

# 1. Vectorize API (Do this once)
api_vector = get_mol2vec_vector(API_SMILES, w2v_model)

if api_vector is None:
    print("❌ Critical Error: Invalid API SMILES.")
    sys.exit(1)

# 2. Loop through Excipients
for exp_name, exp_smiles in EXCIPIENTS_LIST:
    
    # Vectorize Excipient
    exp_vector = get_mol2vec_vector(exp_smiles, w2v_model)
    
    if exp_vector is None:
        print(f"{exp_name:<20} | ❌ Invalid SMILES")
        continue
    
    # 3. Concatenate Features (API + Excipient)
    # The model expects [API_Vector (300), Excipient_Vector (300)] -> Total 600 dims
    feature_vector = np.concatenate([api_vector, exp_vector]).reshape(1, -1)
    
    # 4. Predict
    # Class 0 = Compatible, Class 1 = Incompatible
    prediction_class = rf_model.predict(feature_vector)[0]
    probabilities = rf_model.predict_proba(feature_vector)[0]
    
    # Get probability of the predicted class
    confidence = probabilities[prediction_class] * 100
    
    # Formatting Result
    if prediction_class == 1:
        result_str = "🔴 Incompatible"
    else:
        result_str = "🟢 Compatible"
        
    print(f"{exp_name:<20} | {result_str:<15} | {confidence:.2f}%")

print("-" * 70)
print("Note: Threshold is 50%. Probability indicates model confidence.")


# ======================================================================
# 🧪 COMPATIBILITY PREDICTION REPORT
# 💊 API: Vitamin C (Ascorbic Acid)
# ======================================================================
# Excipient            | Prediction      | Probability
# ----------------------------------------------------------------------
# Mg Stearate          | 🟢 Compatible    | 53.19%
# Fluorinated Amide       | 🔴 Incompatible  | 76.81%
# Cellulose        | 🔴 Incompatible  | 60.99%
# Stearic Acid         | 🔴 Incompatible  | 69.31%
# Mannitol             | 🔴 Incompatible  | 65.83%
# Silicon Dioxide      | 🟢 Compatible    | 57.00%
# ----------------------------------------------------------------------
# Note: Threshold is 50%. Probability indicates mode                  
#                                                 _compatibility>
#Correct Answer:
# Mg -Stearate: Incompatible
# Fluorinated Amide : Incompatible
# Cellulose: Compatible
# Stearic Acid: Compatible
# Mannitol" Compatible
# Silicon Dioxide: Compatible