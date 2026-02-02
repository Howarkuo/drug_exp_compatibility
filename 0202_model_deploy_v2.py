import streamlit as st
import joblib
import pubchempy as pcp
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, DfVec, sentences2vec, MolSentence
from gensim.models import word2vec

# 1. Setup
RDLogger.DisableLog('rdApp.*')
st.title('Drug - Excipient Compatibility (New RF Model)')
st.markdown("---")

# 2. Helper Functions
def safe_mol_convert(smiles):
    if not smiles or not isinstance(smiles, str): return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol: return Chem.AddHs(mol)
    except: return None
    return None

# 3. Load Models (Only need Word2Vec and your NEW Random Forest)
@st.cache_resource
def load_models():
    try:
        # Load the dictionary
        w2v = word2vec.Word2Vec.load('model_300dim.pkl')
        # Load YOUR new model
        rf = joblib.load('0201_my_rf_model.pkl')
        return w2v, rf
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

w2vec_model, model_rf = load_models()

# 4. UI Inputs
col1, col2 = st.columns([1,3])
with col1: option1 = st.selectbox('API Input Type', ['SMILES', 'Name', 'CID'], key='opt1')
with col2: api_input = st.text_input('Enter API', key='txt1')

col3, col4 = st.columns([1,3])
with col3: option3 = st.selectbox('Excipient Input Type', ['SMILES', 'Name', 'CID'], key='opt3')
with col4: exp_input = st.text_input('Enter Excipient', key='txt3')

# 5. Prediction Logic
if st.button('Predict Compatibility', type="primary"):
    if not w2vec_model or not model_rf: st.stop()
    
    # --- Step A: Get SMILES ---
    status = st.empty()
    status.text("Processing: Resolving Structures...")
    
    try:
        # API SMILES
        if option1 == 'SMILES': api_smiles = api_input
        elif option1 == 'Name': api_smiles = pcp.get_compounds(api_input, 'name')[0].isomeric_smiles
        else: api_smiles = pcp.Compound.from_cid(int(api_input)).isomeric_smiles
        
        # Excipient SMILES
        if option3 == 'SMILES': exp_smiles = exp_input
        elif option3 == 'Name': exp_smiles = pcp.get_compounds(exp_input, 'name')[0].isomeric_smiles
        else: exp_smiles = pcp.Compound.from_cid(int(exp_input)).isomeric_smiles
        
    except Exception as e:
        st.error(f"Error finding compound: {e}")
        st.stop()

    # --- Step B: Mol2Vec Feature Engineering ---
    status.text("Processing: Calculating Vectors...")
    
    mol_api = safe_mol_convert(api_smiles)
    mol_exp = safe_mol_convert(exp_smiles)
    
    if not mol_api or not mol_exp:
        st.error("Invalid Chemical Structure.")
        st.stop()
        
    # Vectorize API (100 dims)
    sent_api = MolSentence(mol2alt_sentence(mol_api, 1))
    vec_api = sentences2vec([sent_api], w2vec_model, unseen='UNK')[0]
    
    # Vectorize Excipient (100 dims)
    sent_exp = MolSentence(mol2alt_sentence(mol_exp, 1))
    vec_exp = sentences2vec([sent_exp], w2vec_model, unseen='UNK')[0]
    
    # Combine (200 dims) - This matches your training data format!
    final_features = np.concatenate([vec_api, vec_exp]).reshape(1, -1)
    
    # --- Step C: Prediction ---
    status.text("Processing: Running Random Forest...")
    
    prediction = model_rf.predict(final_features)[0]
    probabilities = model_rf.predict_proba(final_features)[0]
    
    status.empty()
    
    # --- Step D: Display ---
    if prediction == 1:
        st.error(f"**Incompatible** (Probability: {probabilities[1]*100:.2f}%)")
    else:
        st.success(f"**Compatible** (Probability: {probabilities[0]*100:.2f}%)")
        
    st.info("Predicted using new Random Forest model trained on SMOTE-balanced data.")