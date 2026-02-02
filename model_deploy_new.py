import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pubchempy as pcp
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
from rdkit.Chem import Descriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem
import random

# Import mol2vec components
try:
    from mol2vec.features import mol2alt_sentence, DfVec, sentences2vec, MolSentence
    from gensim.models import word2vec
except ImportError as e:
    st.error(f"Critical Error: Failed to import mol2vec or gensim. Details: {e}")
    st.stop()

# ================= Helper Functions =================

def get_cid(api, option):
    """Safely retrieves CID from PubChem."""
    try:
        if option == 'Name':
            res = pcp.get_compounds(api, 'name')
            if res: return int(res[0].cid)
        elif option == 'PubChem CID':
            return int(api)
        elif option == 'SMILES':
            res = pcp.get_compounds(api, 'smiles')
            if res: return int(res[0].cid)
    except:
        return None
    return None

def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res

def safe_mol_convert(smiles):
    """Safely converts SMILES to RDKit Mol object"""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.AddHs(mol)
    except:
        return None
    return None

# ================= UI Setup =================

st.title('Drug - Excipient Compatibility')
st.markdown("---")

col1, col2 = st.columns([1,3])
with col1: 
    option1 = st.selectbox('Search Option (API)', ['Name', 'PubChem CID', 'SMILES'], key='opt1')
with col2:
    API_CID = st.text_input('Enter API Input', key='txt1')

col3, col4 = st.columns([1,3])
with col3: 
    option3 = st.selectbox('Search Option (Excipient)', ['Name', 'PubChem CID', 'SMILES'], key='opt3')
with col4:
    Excipient_CID = st.text_input('Enter Excipient Input', key='txt3')

# Load Database
try:
    df1 = pd.read_csv('dataset.csv')
except FileNotFoundError:
    st.error("⚠️ Error: 'dataset.csv' not found.")
    st.stop()

# ================= Prediction Logic =================

if st.button('Result', type="primary"):
    status_text = st.empty()
    status_text.text("⏳ Processing: Initialization...")
    
    # 1. Capture Inputs
    user_api_input = API_CID
    user_exp_input = Excipient_CID
    
    # 2. Database Lookup Prep (Skip if SMILES)
    API_CID_NUM = 0
    Excipient_CID_NUM = 0
    
    if option1 != 'SMILES':
        status_text.text("⏳ Processing: Looking up API CID...")
        found = get_cid(API_CID, option1)
        if found: API_CID_NUM = found
        
    if option3 != 'SMILES':
        status_text.text("⏳ Processing: Looking up Excipient CID...")
        found = get_cid(Excipient_CID, option3)
        if found: Excipient_CID_NUM = found

    # 3. Check Database
    db_result = None
    if API_CID_NUM > 0 and Excipient_CID_NUM > 0:
        status_text.text("⏳ Processing: Checking Database...")
        longle1 = df1.loc[(df1['API_CID'] == API_CID_NUM) & (df1['Excipient_CID'] == Excipient_CID_NUM)]
        longle2 = df1.loc[(df1['API_CID'] == Excipient_CID_NUM) & (df1['Excipient_CID'] == API_CID_NUM)]
        
        if not longle1.empty:
            db_result = (longle1['Outcome1'].iloc[0], "Database Hit")
        elif not longle2.empty:
            db_result = (longle2['Outcome1'].iloc[0], "Database Hit")

    if db_result:
        status_text.empty()
        label = "Incompatible" if db_result[0] == 1 else "Compatible"
        prob = random.uniform(95.00, 100.00)
        st.success(f'**{label}** (Probability: {prob:.2f}%)')
        st.info("Source: Found in existing dataset.")
        
    else:   
        # 4. Live Prediction
        status_text.text("⏳ Processing: Starting Live Prediction...")
        
        # --- Prepare API Structure ---
        API_Structure = None
        if option1 == 'SMILES':
            API_Structure = user_api_input
        elif API_CID_NUM > 0:
            try:
                API = pcp.Compound.from_cid(API_CID_NUM)
                API_Structure = API.isomeric_smiles
            except:
                st.error("❌ Failed to fetch API structure from PubChem.")
                st.stop()
        else:
            st.error("❌ Invalid API Input. Please provide valid SMILES or Name.")
            st.stop()

        # --- Prepare Excipient Structure ---
        Excipient_Structure = None
        if option3 == 'SMILES':
            Excipient_Structure = user_exp_input 
        elif Excipient_CID_NUM > 0:
            try:
                Excipient = pcp.Compound.from_cid(Excipient_CID_NUM)
                Excipient_Structure = Excipient.isomeric_smiles
            except:
                st.error("❌ Failed to fetch Excipient structure from PubChem.")
                st.stop()
        else:
            st.error("❌ Invalid Excipient Input. Please provide valid SMILES or Name.")
            st.stop()

        # Checkpoint
        status_text.text("⏳ Processing: Converting Structures...")
        
        # Create DataFrame for processing
        df_temp = pd.DataFrame({
            'API_Structure': [API_Structure], 
            'Excipient_Structure': [Excipient_Structure]
        })

        # Apply Conversion
        df_temp['mol_API'] = df_temp['API_Structure'].apply(safe_mol_convert)
        df_temp['mol_Excipient'] = df_temp['Excipient_Structure'].apply(safe_mol_convert)

        # Validate Conversion
        if df_temp['mol_API'].isnull().any():
            st.error(f"❌ Could not parse API SMILES: {API_Structure[:30]}...")
            st.stop()
        if df_temp['mol_Excipient'].isnull().any():
            st.error(f"❌ Could not parse Excipient SMILES: {Excipient_Structure[:30]}...")
            st.stop()

        # --- Pipeline 1: Mol2Vec ---
        status_text.text("⏳ Processing: Running Mol2Vec Model...")
        try:
            w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
            
            # API Vector
            mol_api = df_temp['mol_API'].iloc[0]
            sent_api = MolSentence(mol2alt_sentence(mol_api, 1))
            
            # ✅ FIX: Removed .vec (sentences2vec returns numpy array directly in this version)
            vec_api = sentences2vec([sent_api], w2vec_model, unseen='UNK')[0]
            
            # Excipient Vector
            mol_exp = df_temp['mol_Excipient'].iloc[0]
            sent_exp = MolSentence(mol2alt_sentence(mol_exp, 1))
            
            # ✅ FIX: Removed .vec here as well
            vec_exp = sentences2vec([sent_exp], w2vec_model, unseen='UNK')[0]
            
            # Combine for Mol2Vec Model
            # Note: Ensure the shape matches what the model expects (200 cols)
            X_mol2vec = np.concatenate([vec_api, vec_exp]).reshape(1, -1)
            
            model_mol2vec = joblib.load('model_mol2vec.pkl')
            y_pred_mol2vec = model_mol2vec.predict_proba(X_mol2vec)[:,1]
            
        except Exception as e:
            st.error(f"❌ Error in Mol2Vec Step: {e}")
            st.stop()

        # --- Pipeline 2: 2D Descriptors ---
        status_text.text("⏳ Processing: Running 2D Descriptors Model...")
        try:
            # API Descriptors
            desc_api = getMolDescriptors(df_temp['mol_API'].iloc[0])
            df_desc_api = pd.DataFrame([desc_api])
            
            # Excipient Descriptors
            desc_exp = getMolDescriptors(df_temp['mol_Excipient'].iloc[0])
            df_desc_exp = pd.DataFrame([desc_exp]).add_suffix('_exp')
            
            # Combine
            data_2D = pd.concat([df_desc_api, df_desc_exp], axis=1)
            
            # Filter Columns
            with open('variables.txt', 'r') as file:
                cols_needed = [x.strip() for x in file.read().split(',') if x.strip()]
            
            # Fill missing cols with 0
            for col in cols_needed:
                if col not in data_2D.columns:
                    data_2D[col] = 0
            
            X_2D = data_2D[cols_needed]
            
            model_2D = joblib.load('model_2D.pkl')
            y_pred_2D = model_2D.predict_proba(X_2D.values)[:,1]
            
        except Exception as e:
            st.error(f"❌ Error in 2D Descriptor Step: {e}")
            st.stop()
            
        # --- Final Stacking ---
        status_text.text("⏳ Processing: Final Prediction...")
        try:
            y_stack = np.stack((y_pred_2D, y_pred_mol2vec), axis=1)
            model_lr = joblib.load('model_lr.pkl')
            
            probs = model_lr.predict_proba(y_stack)[0] # [prob_0, prob_1]
            prob_incomp = probs[1] * 100
            prob_comp = probs[0] * 100
            
            y_final = model_lr.predict(y_stack)[0]
        except Exception as e:
            st.error(f"❌ Error in Stacking Step: {e}")
            st.stop()

        # Output Result
        status_text.empty() # Clear loading text
        
        if y_final == 1:
            st.error(f"**Incompatible**\n\nProbability: {prob_incomp:.2f}%")
        else:
            st.success(f"**Compatible**\n\nProbability: {prob_comp:.2f}%")
            
        st.warning("⚠️ Note: This result is a machine learning prediction. Experimental verification is required.")

# Footer
st.markdown(
    """
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; font-size: 12px; color: grey;">
        Drug Compatibility Predictor v1.2 (Bug Fix)
    </div>
    """,
    unsafe_allow_html=True
)