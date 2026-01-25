import pandas as pd
import numpy as np
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
import traceback # 用來印出詳細錯誤

# ================= 設定區 =================
INPUT_FILE = '2024_Drug_compatibility_dataset.xlsx' 
COL_API = 'API_Smiles'         
# =========================================

def debug_process():
    print(f"📂 Reading {INPUT_FILE} (First 5 rows only)...")
    try:
        df = pd.read_excel(INPUT_FILE).head(5) # 只取前 5 筆
    except Exception as e:
        print(e)
        return

    print("🧠 Loading Mol2vec model...")
    w2v_model = word2vec.Word2Vec.load('model_300dim.pkl')

    print("\n🕵️‍♂️ --- STARTING DEBUGGING ---")
    
    # 我們不使用 apply，而是直接跑迴圈，這樣可以針對每一行除錯
    for index, row in df.iterrows():
        smiles = row[COL_API]
        print(f"\nProcessing Row {index}: {smiles}")
        
        try:
            # 1. 轉成 RDKit 分子
            mol = Chem.MolFromSmiles(str(smiles))
            if not mol:
                print("❌ [Fail] RDKit could not parse SMILES")
                continue
            print("   ✅ RDKit Parsed")

            # 2. 加氫
            mol = Chem.AddHs(mol)
            print("   ✅ AddHs Done")
            
            # 3. 產生句子 (這是最可能出錯的地方)
            print("   👉 Attempting mol2alt_sentence...")
            sentence_data = mol2alt_sentence(mol, 1)
            print(f"   ✅ mol2alt_sentence Done. Words: {len(sentence_data)}")

            # 4. 包裝成 MolSentence
            sentence = MolSentence(sentence_data)
            print("   ✅ MolSentence Object Created")
            
            # 5. 轉向量
            print("   👉 Attempting sentences2vec...")
            # 注意：這裡可能回傳空值或格式不對
            vec_list = sentences2vec([sentence], w2v_model, unseen='UNK')
            
            if len(vec_list) == 0:
                print("   ❌ [Fail] sentences2vec returned empty list!")
                continue

            vec = vec_list[0]
            print(f"   ✅ SUCCESS! Vector shape: {vec.vec.shape}")

        except Exception as e:
            print(f"   🔥 CRASHED WITH ERROR: {e}")
            print("   --- Traceback details ---")
            traceback.print_exc()
            print("   -------------------------")
            break # 只要抓到一個錯誤就停下來，不要洗版

if __name__ == "__main__":
    debug_process()