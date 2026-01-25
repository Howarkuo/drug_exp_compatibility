import pandas as pd
import numpy as np
import sys
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec

# ================= 設定區 (Configuration) =================
INPUT_FILE = '2024_Drug_compatibility_dataset.xlsx' 
OUTPUT_FILE = '2024_Drug_compatibility_dataset.xlsx_train_data_vectors_final.csv'

# 欄位對應
COL_API_CID = 'API_CID'
COL_EXP_CID = 'Excipient_CID'
COL_API_SMILES = 'API_Smiles'         
COL_EXP_SMILES = 'Excipient_Smiles'   
COL_LABEL = 'Outcome (1: incompatible; 0 compatible)' 
# =========================================================

def process_final():
    print(f" Reading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f" Error: '{INPUT_FILE}' not found.")
        return

    print(" Loading Mol2vec model...")
    try:
        w2v_model = word2vec.Word2Vec.load('model_300dim.pkl')
    except:
        print(" Error: 'model_300dim.pkl' not found.")
        return

    def smiles_to_vector(smiles):
        try:
            # 1. 基礎處理
            mol = Chem.MolFromSmiles(str(smiles))
            if not mol: return None
            mol = Chem.AddHs(mol)
            sentence = MolSentence(mol2alt_sentence(mol, 1))
            
            # 2. 轉向量
            # sentences2vec 回傳的是一個矩陣 (n_sentences, n_features)
            # 我們只傳入一個句子，所以取第 0 個結果
            vector_matrix = sentences2vec([sentence], w2v_model, unseen='UNK')
            
            # ⚠️ 關鍵修正在此：直接回傳，不要加 .vec
            return vector_matrix[0] 
            
        except Exception:
            return None

    print("  Converting SMILES to Vectors...")
    df['vec_API'] = df[COL_API_SMILES].apply(smiles_to_vector)
    df['vec_EXP'] = df[COL_EXP_SMILES].apply(smiles_to_vector)

    # 移除失敗資料
    df_clean = df.dropna(subset=['vec_API', 'vec_EXP']).reset_index(drop=True)
    print(f"   Success Rate: {len(df_clean)} / {len(df)}")
    
    if len(df_clean) == 0:
        print(" CRITICAL ERROR: No data converted. Check logic again.")
        return

    print("Formatting final dataset...")
    
    # 展開 API 向量
    # 使用 list comprehension 確保 numpy array 堆疊正確
    api_matrix = np.array(df_clean['vec_API'].tolist())
    api_df = pd.DataFrame(api_matrix, columns=[f'API_{i}' for i in range(api_matrix.shape[1])])
    
    # 展開 Excipient 向量
    exp_matrix = np.array(df_clean['vec_EXP'].tolist())
    exp_df = pd.DataFrame(exp_matrix, columns=[f'EXP_{i}' for i in range(exp_matrix.shape[1])])
    
    # 合併 Meta Data (CID + Label)
    meta_data = df_clean[[COL_API_CID, COL_EXP_CID, COL_LABEL]].copy()
    meta_data = meta_data.rename(columns={COL_LABEL: 'Label'})
    
    final_df = pd.concat([meta_data, api_df, exp_df], axis=1)
    
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n MISSION COMPLETE!")
    print(f"   Saved file: {OUTPUT_FILE}")
    print(f"   Shape: {final_df.shape}")
    print("   Next Step: Send this file to Eddie for SMOTE training.")

if __name__ == "__main__":
    process_final()