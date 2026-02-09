import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# 1. 定義測試分子
# Vitamin C (Ascorbic Acid)
# Mg Stearate (含有 [Mg+2] 的硬脂酸鎂)
molecules = {
    "Vitamin C": "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O",
    "Mg Stearate": "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"
}

print(f"⚙️ Calculating Mordred descriptors for {len(molecules)} molecules...")

# 2. 轉換為 RDKit 物件
mols = [Chem.MolFromSmiles(smi) for smi in molecules.values()]

# 3. 設定計算機 (只算 2D，忽略 3D)
calc = Calculator(descriptors, ignore_3D=True)

# 4. 執行計算
# n_proc=1 避免並行運算的 overhead，小數據單核更快
df_raw = calc.pandas(mols, nproc=1, quiet=True)

# 處理非數值錯誤 (填 0)
df = df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

# 加上名字當 Index
df.index = molecules.keys()

# 5. 篩選我們最關心的「相容性關鍵特徵」
# 我們想看：酸性基團、鹼性基團、鎂原子數、電荷相關特徵
key_features = [
    'nAcid',      # 酸性基團數 (Vitamin C 應該高)
    'nBase',      # 鹼性基團數
    'nMg',        # 鎂原子數 (這就是 Mol2Vec 看不到的關鍵!)
    'MW',         # 分子量
    'TopoPSA',    # 極性表面積 (吸濕性指標)
    'nRot',       # 可旋轉鍵 (Mg Stearate 的長鏈應該很多)
    'GATS1c'      # 電荷相關拓撲特徵 (只是範例，Mordred 有很多這類特徵)
]

# 嘗試找出存在的欄位 (有些版本名稱可能微調)
available_cols = [c for c in key_features if c in df.columns]

# 如果找不到具體的 'nMg'，我們搜尋所有跟 'Mg' 有關的欄位
mg_cols = [c for c in df.columns if 'Mg' in c]
final_cols = list(set(available_cols + mg_cols))

print("\n" + "="*60)
print("🧪 Mordred Descriptor Comparison")
print("="*60)
print(df[final_cols].T)  # 轉置表格方便閱讀
print("="*60)

# 6. 自動判斷
print("\n💡 自動診斷結果：")

try:
    vit_acid = df.loc['Vitamin C', 'nAcid']
    mg_atom = df.loc['Mg Stearate', 'nMg'] if 'nMg' in df.columns else 0
    
    if vit_acid > 0:
        print(f"✅ 成功偵測到 Vitamin C 的酸性 (nAcid = {vit_acid})")
    else:
        print("❌ 未偵測到 Vitamin C 的酸性")

    if mg_atom > 0:
        print(f"✅ 成功偵測到 Mg Stearate 的鎂離子 (nMg = {mg_atom})")
        print("   這證明了 Mordred 比 Mol2Vec 更能捕捉無機鹽類特徵！")
    else:
        print("❌ 未偵測到 Mg Stearate 的鎂離子")

except Exception as e:
    print(f"⚠️ 診斷時發生錯誤: {e}")


# ============================================================
# 🧪 Mordred Descriptor Comparison
# ============================================================
#           Vitamin C  Mg Stearate
# nBase      0.000000     0.000000
# GATS1c     1.449930     0.175450
# nRot       2.000000    32.000000
# nAcid      0.000000     2.000000
# MW       176.032088   590.512452
# TopoPSA  107.220000    80.260000
# ============================================================