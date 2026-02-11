import pandas as pd
import numpy as np
import os
import joblib
import pubchempy as pcp
import cirpy
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 關閉警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration
# ==========================================
# 輸入與輸出
INPUT_FILE = '../0210_Compatibility_Testset_162.xlsx'  # 原始 Excel (只有 CID)
OUTPUT_FILE = '0210_mordred_rf_prediction_results.csv'

# 模型與特徵列表 (剛剛訓練好的)
MODEL_FILE = '0210_modelered_rf_model.pkl'
FEATURE_FILE = '0210_mordred_features_list.pkl'

# ==========================================
# 2. Helper Functions
# ==========================================

def get_smiles_robust(cid):
    """從 PubChem 或 CIRpy 抓取 SMILES"""
    cid = int(cid)
    smi = None
    try:
        props = pcp.get_properties('IsomericSMILES', cid)
        if props and 'IsomericSMILES' in props[0]:
            smi = props[0]['IsomericSMILES']
    except:
        pass
    
    if not smi:
        try:
            c = pcp.Compound.from_cid(cid)
            if c.inchikey:
                smi = cirpy.resolve(c.inchikey, 'smiles')
        except:
            pass
    return smi

def generate_mordred_features(smiles_list, prefix):
    """計算 Mordred 特徵"""
    print(f"⚙️ Generating {prefix} features...")
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    
    # 過濾無效分子
    valid_mols = [m for m in mols if m is not None]
    if not valid_mols: return pd.DataFrame()
    
    calc = Calculator(descriptors, ignore_3D=True)
    
    try:
        # 這裡開啟 quiet=False 讓您看到進度
        df_features = calc.pandas(valid_mols, nproc=1, quiet=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 對齊回原始長度
    final_df = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_features.columns)
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    final_df.iloc[valid_indices] = df_features.values
    
    return final_df.add_prefix(prefix)

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("📂 Loading resources...")
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURE_FILE):
        print("❌ Model files not found. Please run training script first."); exit()

    # 1. Load Model & Feature List
    rf_model = joblib.load(MODEL_FILE)
    required_features = joblib.load(FEATURE_FILE)
    print(f"✅ Loaded Model & Feature List ({len(required_features)} features).")

    # 2. Load Data & Fetch SMILES
    print(f"📂 Reading data: {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    
    # 重新命名 Label 欄位
    target_col = 'Outcome (1: incompatible; 0 compatible)'
    if target_col in df.columns:
        df = df.rename(columns={target_col: 'Label'})

    # 抓取 SMILES
    print("🌍 Fetching SMILES (PubChem + CIRpy)...")
    tqdm.pandas()
    df = df.dropna(subset=['API_CID', 'Excipient_CID'])
    df['API_SMILES'] = df['API_CID'].progress_apply(get_smiles_robust)
    df['EXP_SMILES'] = df['Excipient_CID'].progress_apply(get_smiles_robust)
    
    # 移除抓不到的資料
    df_clean = df.dropna(subset=['API_SMILES', 'EXP_SMILES']).copy()
    print(f"📊 Valid pairs for prediction: {len(df_clean)} / {len(df)}")

    # 3. Generate Features (Full Set)
    df_api = generate_mordred_features(df_clean['API_SMILES'].tolist(), "API_")
    df_exp = generate_mordred_features(df_clean['EXP_SMILES'].tolist(), "EXP_")
    
    X_raw = pd.concat([df_api, df_exp], axis=1)
    
    # 4. ALIGN FEATURES (關鍵步驟！)
    # 強制將特徵對齊到訓練時的 2738 個欄位，缺補 0，多刪除
    print(f"🔗 Aligning features (Raw: {X_raw.shape[1]} -> Required: {len(required_features)})...")
    X_final = X_raw.reindex(columns=required_features, fill_value=0)
    
    # 5. Predict
    print("🔮 Predicting...")
    preds = rf_model.predict(X_final)
    probs = rf_model.predict_proba(X_final)[:, 1]
    
    # 6. Report & Save
    df_clean['RF_Mordred_Pred'] = preds
    df_clean['RF_Mordred_Prob'] = probs
    
    if 'Label' in df_clean.columns:
        y_true = df_clean['Label'].astype(int)
        acc = accuracy_score(y_true, preds)
        
        print("\n" + "="*60)
        print(f"🧪 RF + Mordred Test Report (Accuracy: {acc:.4f})")
        print("="*60)
        print("Confusion Matrix:")
        cm = confusion_matrix(y_true, preds)
        print(cm)
        print("-" * 30)
        print("Classification Report:")
        print(classification_report(y_true, preds))

        # 計算 Critical Recall (Class 1)
        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
            recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"\n🔥 Critical Recall (Class 1 - Incompatible): {recall_class1:.4f}")

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")