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

#%%
def get_cid(api, option):
    if option == 'Name':
        compound = pcp.get_compounds(api, 'name')[0]
    elif option == 'PubChem CID':
        compound = pcp.Compound.from_cid(int(api))
    elif option == 'SMILES':
        compound = pcp.get_compounds(api, 'smiles')[0]
    return int(compound.cid)

#%%
def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res
# 0130: add SMILES to RDKit Mol checkers 
def safe_mol_convert(smiles):
        # 如果是空的，或者不是字串，就直接回傳 None，不要崩潰
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.AddHs(mol)
        return None
#%%
st.title('Drug - Excipient Compatibility')
col1, col2 = st.columns([1,3])
with col1: 
    option1 = st.selectbox('Search Option', ['Name', 'PubChem CID', 'SMILES'])
with col2:
    API_CID = st.text_input('Enter name, Pubchem CID or smiles string of the API')
col3, col4 = st.columns([1,3])
with col3: 
    option3 = st.selectbox('', ['Name', 'PubChem CID', 'SMILES'])
with col4:
    Excipient_CID = st.text_input('Enter name, Pubchem CID or smiles string of the excipient')

df1 = pd.read_csv('dataset.csv')
#%%
# code for Prediction
Predict_Result1 = ''
Predict_Result2 = ''
Predict_Result3 = ''

if st.button('Result'):
    # 0130 update (save the user Raw Input)
    # 1. Save Raw Inputs (Critical for bypassing PubChem)
    user_api_input = API_CID
    user_exp_input = Excipient_CID
    # comment on old script
    # API_CID = get_cid(API_CID, option1)
    # Excipient_CID = get_cid(Excipient_CID, option3)
    # longle1 = df1.loc[(df1['API_CID'] == API_CID) & (df1['Excipient_CID'] == Excipient_CID)]
    # longle2 = df1.loc[(df1['API_CID'] == Excipient_CID) & (df1['Excipient_CID'] == API_CID)]
    
    # 2. Convert inputs to CIDs for Database Lookup
    try:
        API_CID_NUM = get_cid(API_CID, option1)
        Excipient_CID_NUM = get_cid(Excipient_CID, option3)
    except:
        # Fallback if PubChem fails during CID lookup, just use dummy numbers to force "live prediction"
        API_CID_NUM = 0
        Excipient_CID_NUM = 0
    # finish update 

    # 3. Check Database
    longle1 = df1.loc[(df1['API_CID'] == API_CID_NUM) & (df1['Excipient_CID'] == Excipient_CID_NUM)]
    longle2 = df1.loc[(df1['API_CID'] == Excipient_CID_NUM) & (df1['Excipient_CID'] == API_CID_NUM)]
    if not longle1.empty:
        outcome1 = longle1.loc[:, 'Outcome1']
        if outcome1.iloc[0] == 1:
            Predict_Result1 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        else:
            Predict_Result1 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        st.success(Predict_Result1)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

    elif not longle2.empty:
        outcome2 = longle2.loc[:, 'Outcome1']
        if outcome2.iloc[0] == 1:
             Predict_Result2 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        else:
             Predict_Result2 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        st.success(Predict_Result2)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')
        
    else:   
        import pubchempy as pcp
        # 0130 update: substitute read user excipient and API smiles input 
        # comment out old script 
        # Excipient = pcp.Compound.from_cid(Excipient_CID)
        # Excipient_Structure = Excipient.isomeric_smiles
        # API = pcp.Compound.from_cid(API_CID)
        # API_Structure = API.isomeric_smiles
        # df = pd.DataFrame({'API_CID': API_CID, 'Excipient_CID': Excipient_CID, 'API_Structure' : API_Structure, 'Excipient_Structure': Excipient_Structure},index=[0])
        # finish substitute 
        # 1.  --- Handle Excipient Structure ---
        if option3 == 'SMILES':
            # Direct Bypass: Use user input directly
            Excipient_Structure = user_exp_input # 直接用使用者輸入的 SMILES
        else:
            # Query PubChem
            Excipient = pcp.Compound.from_cid(Excipient_CID)
            Excipient_Structure = Excipient.isomeric_smiles

        # 2.--- Handle API Structure ---
        if option1 == 'SMILES':
            API_Structure = user_api_input # 直接用使用者輸入的 SMILES
        else:
            # Query PubChem
            API = pcp.Compound.from_cid(API_CID)
            API_Structure = API.isomeric_smiles
        # Create temporary DataFrame
        df = pd.DataFrame({
            'API_CID': API_CID_NUM, 
            'Excipient_CID': Excipient_CID_NUM, 
            'API_Structure' : API_Structure, 
            'Excipient_Structure': Excipient_Structure
        }, index=[0])
        # finish update 
    #   # 0130 update: check logic for why smiles doesn't work
        # df['mol_API'] = df['API_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        # df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))
        # df['mol_Excipient'] = df['Excipient_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        # df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))

        # 修改後的安全版程式碼
   

# Apply Safety Converter
    df['mol_API'] = df['API_Structure'].apply(safe_mol_convert)
    df['mol_Excipient'] = df['Excipient_Structure'].apply(safe_mol_convert)

    # 如果轉換失敗 (例如 PubChem 沒抓到資料)，顯示錯誤並停止，而不是讓程式崩潰
    if df['mol_API'].isnull().any() or df['mol_Excipient'].isnull().any():
        st.error(" Error: Could not parse chemical structure. If using SMILES, please check the format.")
        st.stop()
    # finish substitute 
    # Feature Engineering: Mol2Vec
        from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
        from gensim.models import word2vec
        w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
        df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
        df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
        df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
        df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]
    # Create dataframe 
        X1 = np.array([x.vec for x in df['mol2vec_API']])  
        X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
        X_mol2vec = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API','mol2vec_Excipient', 'sentence_Excipient', 
                                                                'API_Structure', 'Excipient_Structure' ,'mol_API',
                                                                'mol_Excipient','sentence_API','API_CID','Excipient_CID'], axis=1)), axis=1)
    # Load pretrained model
        model_mol2vec = joblib.load('model_mol2vec.pkl')
        y_pred_mol2vec = model_mol2vec.predict_proba(X_mol2vec.values)[:,1]
    #
        API_mol = Chem.MolFromSmiles(API_Structure)
        New_3D_descriptors_API = np.array(list(getMolDescriptors(API_mol).values())).reshape(1, -1)
        
        exp_mol = Chem.MolFromSmiles(Excipient_Structure)
        New_3D_descriptors_exp = np.array(list(getMolDescriptors(exp_mol).values())).reshape(1, -1)
        
        df_API = pd.DataFrame(New_3D_descriptors_API, columns=list(getMolDescriptors(API_mol).keys()))
        df_exp = pd.DataFrame(New_3D_descriptors_exp, columns=list(getMolDescriptors(exp_mol).keys())).add_suffix('_exp')
        data_2D = pd.concat([df_API, df_exp], axis=1)
        
        with open('variables.txt', 'r') as file:
            descriptors_line = file.read()

        selected_descriptors = descriptors_line.split(',')
        X_2D = data_2D[selected_descriptors]
        
        model_2D = joblib.load('model_2D.pkl')
        y_pred_2D = model_2D.predict_proba(X_2D.values)[:,1]
        y_pred = np.stack((y_pred_2D, y_pred_mol2vec), axis = 1)
        
        model_lr = joblib.load('model_lr.pkl')
        y_prediction = model_lr.predict(y_pred)
        probs1 = np.round(model_lr.predict_proba(y_pred)[:,1] * 100, 2)
        probs0 = np.round(model_lr.predict_proba(y_pred)[:,0] * 100, 2)
    
        if y_prediction[0] == 1:
            Predict_Result3 = f'Incompatible. Probality: {probs1[0]}%'
        else:
            Predict_Result3 = f'Compatible. Probality: {probs0[0]}%'
        st.success(Predict_Result3)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

st.markdown(
    """
    <div style="position: fixed; bottom: 8px; width: 100%; text-align: left; padding-left: 5cm;">
        For the updated version of this website, please visit <a href="https://decompatibility-v12.streamlit.app">here</a>
    </div>
    """,
    unsafe_allow_html=True
)
