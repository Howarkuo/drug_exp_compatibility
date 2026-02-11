import pandas as pd
import numpy as np
import os
import joblib
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. 檔案路徑設定
# ==========================================
TRAIN_FILE = '2024_Drug_compatibility_dataset.xlsx'
TEST_FILE = 'Mordered/0210_mordred_rf_prediction_results.csv' # 使用您上一部跑完、有 SMILES 的結果檔
FEATURE_FILE = 'Mordered/0210_mordred_features_list.pkl'      # 對齊特徵用

# ==========================================
# 2. 特徵計算函數 (與訓練時相同)
# ==========================================
def generate_mordred_features(smiles_list, prefix):
    print(f"⚙️ Generating {prefix} features...")
    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None for s in smiles_list]
    valid_mols = [m for m in mols if m is not None]
    
    calc = Calculator(descriptors, ignore_3D=True)
    df_features = calc.pandas(valid_mols, nproc=1, quiet=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    final_df = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_features.columns)
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    final_df.iloc[valid_indices] = df_features.values
    return final_df.add_prefix(prefix)

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    
    print("📂 Loading Required Features List...")
    if not os.path.exists(FEATURE_FILE):
        print("❌ Feature list not found!"); exit()
    required_features = joblib.load(FEATURE_FILE)

    # --- A. 處理 Test Set (162筆) ---
    print(f"\n📂 Loading Test Set: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)
    df_test_api = generate_mordred_features(df_test['API_SMILES'].tolist(), "API_")
    df_test_exp = generate_mordred_features(df_test['EXP_SMILES'].tolist(), "EXP_")
    X_test_raw = pd.concat([df_test_api, df_test_exp], axis=1)
    X_test = X_test_raw.reindex(columns=required_features, fill_value=0)
    
    # --- B. 處理 Train Set (3544筆) ---
    print(f"\n📂 Loading Train Set: {TRAIN_FILE}")
    df_train = pd.read_excel(TRAIN_FILE)
    # 這裡會跑個 10 分鐘，請耐心等候
    df_train_api = generate_mordred_features(df_train['API_Smiles'].tolist(), "API_")
    df_train_exp = generate_mordred_features(df_train['Excipient_Smiles'].tolist(), "EXP_")
    X_train_raw = pd.concat([df_train_api, df_train_exp], axis=1)
    X_train = X_train_raw.reindex(columns=required_features, fill_value=0)

    # # --- C. 合併並標記來源 ---
    # print("\n🔗 Combining and Scaling Data...")
    # X_train['Dataset'] = 'Training Data (3544 items)'
    # X_test['Dataset'] = 'Validation Data (162 items)'
    
    # X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    # labels = X_combined['Dataset'].values
    
    # # 移除 Dataset 標籤並進行標準化 (UMAP 必須先標準化)
    # X_features = X_combined.drop(columns=['Dataset'])
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_features)

    # # --- D. 執行 UMAP 降維 ---
    # print("🗺️ Running UMAP Dimension Reduction (Transforming ~2700D to 2D)...")
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    # embedding = reducer.fit_transform(X_scaled)

    # --- C. 分開標準化 (嚴謹做法: Scaler 只 fit Train) ---
    print("\n🔗 Scaling Data (Fit on Train, Transform on Test)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # 只做 transform

    # --- D. 執行 UMAP 降維 (嚴謹做法: UMAP 只 fit Train) ---
    print("🗺️ Running UMAP Dimension Reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    
    # 1. 先把訓練集降維 (Fit + Transform)
    embedding_train = reducer.fit_transform(X_train_scaled)
    
    # 2. 把測試集投影到剛建好的空間中 (Transform Only)
    embedding_test = reducer.transform(X_test_scaled)
    
    # 3. 把座標合併起來準備畫圖
    embedding = np.vstack((embedding_train, embedding_test))
    
    # 建立標籤
    labels = ['Training Data (3544 items)'] * len(X_train) + ['Validation Data (162 items)'] * len(X_test)

    # --- E. 繪製精美散佈圖 ---
    print("🎨 Plotting visualization...")
    plt.figure(figsize=(10, 8))
    
    # 使用 seaborn 繪製，調整點的大小與透明度凸顯 162 筆測試集
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=labels,
        palette={'Training Data (3544 items)': '#B0BEC5', 'Validation Data (162 items)': '#E53935'},
        alpha=0.7,
        s=[20 if l == 'Training Data (3544 items)' else 80 for l in labels],
        edgecolor=None
    )

    plt.title('UMAP Chemical Space: Training vs. Validation (162 items)', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Dataset Origin', fontsize=10, title_fontsize=12)
    plt.tight_layout()

    # 存檔
    output_img = 'UMAP_Train_vs_Test_Distribution.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ Success! Map saved to: {output_img}")