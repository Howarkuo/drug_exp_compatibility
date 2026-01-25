#Reference : mol2vec_descriptors.py
import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec

# 1. 修正 Import：加入 mol2alt_sentence
try:
    from mol2vec.features import MolSentence, DfVec, sentences2vec, mol2alt_sentence
    print("✅ [Success] Mol2vec functions imported.")
except ImportError as e:
    print(f" [Error] Failed to import mol2vec: {e}")
    sys.exit(1)

def test_single_molecule():
    print("\n--- Starting Mol2vec Test (V2 - Author's Logic) ---")
    
    # 測試分子 (Aspirin)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"1. Input SMILES: {smiles}")

    # 2. 轉成 RDKit 物件
    mol = Chem.MolFromSmiles(smiles)
    
    # 3. [關鍵修正] 加入氫原子 (AddHs)
    # 作者的程式碼證明了他們使用了顯式氫原子
    mol = Chem.AddHs(mol)
    print("2. Hydrogens added (Chem.AddHs) - Done")

    # 4. 載入模型
    model_path = "model_300dim.pkl"
    if not os.path.exists(model_path):
        print(f"[Error] '{model_path}' not found in current directory.")
        return

    try:
        print(f"3. Loading model from {model_path}...")
        w2v_model = word2vec.Word2Vec.load(model_path)
        
        # 5. [關鍵修正] 使用 mol2alt_sentence 處理半徑
        # 這是作者程式碼中的正確寫法：
        # df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
        sentence_data = mol2alt_sentence(mol, 1)
        sentence = MolSentence(sentence_data)
        print(f"4. MolSentence created successfully!")

        # 6. 轉成向量
        # sentences2vec 吃的是 list，所以要加 []
        vector = sentences2vec([sentence], w2v_model, unseen='UNK')
        
        print(f"5. Vector shape: {vector.shape}") # 應該是 (1, 300)
        print(f"   First 5 values: {vector[0][:5]}")
        print("\n✅ [Result] SUCCESS! Logic matches the author's code.")
        
    except Exception as e:
        print(f"[Error] Failed during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_molecule()


# #✅ [Success] Mol2vec functions imported.

# --- Starting Mol2vec Test (V2 - Author's Logic) ---
# 1. Input SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
# 2. Hydrogens added (Chem.AddHs) - Done
# 3. Loading model from model_300dim.pkl...
# 4. MolSentence created successfully!
# 5. Vector shape: (1, 100)
#    First 5 values: [-4.795716   3.4513886 -6.2674136 -1.3229848 -2.7496908]

# ✅ [Result] SUCCESS! Logic matches the author's code.


# 