import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# ==========================================
# 1. Define Molecules for Hypothesis Testing
# ==========================================
# Hypothesis: Mordred should detect the Mg atom and ionic nature 
# distinguishing Stearic Acid from its Magnesium salt.
molecules = {
    "Stearic_Acid": "CCCCCCCCCCCCCCCCCC(=O)O",
    "Mg_Stearate": "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"
}

print("⚙️  Running Feature Hunter: Stearic Acid vs. Mg Stearate...")

# ==========================================
# 2. Calculate Mordred Descriptors
# ==========================================
# Convert SMILES to RDKit Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in molecules.values()]

# Initialize Calculator (2D only, ignore 3D for speed/stability)
calc = Calculator(descriptors, ignore_3D=True)

# Calculate features (nproc=1 for Windows stability)
df = calc.pandas(mols, nproc=1, quiet=True)

# Clean data: Coerce errors to numeric and fill NaNs with 0
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df.index = molecules.keys()

# ==========================================
# 3. Identify Distinctive Features
# ==========================================
# Logic: Find features where Mg_Stearate has a value, but Stearic_Acid is 0.
# This isolates the effect of the Magnesium ion.

print("\n" + "="*80)
print("🕵️‍♂️  Unique Features detected in Mg Stearate (absent in Stearic Acid)")
print("="*80)
print(f"{'Feature Name':<25} | {'Stearic Ac.':<12} | {'Mg Stearate':<12} | {'Interpretation'}")
print("-" * 80)

found_mg_signal = False
count = 0

for col in df.columns:
    val_acid = df.loc["Stearic_Acid", col]
    val_salt = df.loc["Mg_Stearate", col]
    
    # # Condition: Distinctive feature present in Salt but not in Acid
    # if val_acid == 0 and val_salt > 0:
        
    #     # Determine likely physical meaning for annotation
    #     interpretation = ""
    #     if 'Z' in col: interpretation = "Atomic No. (Mg=12)"
    #     elif 'PEOE' in col: interpretation = "Electrostatics/Charge"
    #     elif 'Acid' in col: interpretation = "Stoichiometry"
    #     elif 'Mg' in col or 'Metal' in col: interpretation = "Metal property"
        
    #     # Print significant features
    #     print(f"{col:<25} | {val_acid:<12.4f} | {val_salt:<12.4f} | {interpretation}")
    #     count += 1
        
    #     # Check if we found direct metal/atomic evidence
    #     if 'Z' in col or 'Metal' in col or 'Mg' in col:
    #         found_mg_signal = True
    import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors

# ==========================================
# 1. Define Molecules
# ==========================================
molecules = {
    "Stearic_Acid": "CCCCCCCCCCCCCCCCCC(=O)O",
    "Mg_Stearate": "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"
}

print("⚙️  Running Feature Hunter v2 (Difference Mode)...")

# ==========================================
# 2. Calculate Mordred Descriptors
# ==========================================
mols = [Chem.MolFromSmiles(smi) for smi in molecules.values()]
calc = Calculator(descriptors, ignore_3D=True)

# nproc=1 for Windows safety
df = calc.pandas(mols, nproc=1, quiet=True)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df.index = molecules.keys()

# ==========================================
# 3. Compare Differences
# ==========================================
print("\n" + "="*90)
print("🕵️‍♂️  Top Differences: Stearic Acid vs. Mg Stearate")
print("="*90)
print(f"{'Feature':<25} | {'Stearic Ac.':<12} | {'Mg Stearate':<12} | {'Diff Ratio':<8} | {'Note'}")
print("-" * 90)

count = 0
significant_features = []

# for col in df.columns:
#     val_acid = df.loc["Stearic_Acid", col]
#     val_salt = df.loc["Mg_Stearate", col]
    
#     # 邏輯修正：只要數值差異超過 10% 或是絕對值差異很大，就顯示出來
#     diff = abs(val_salt - val_acid)
    
#     # 避免除以零
#     if val_acid == 0:
#         ratio = 999.0 if val_salt > 0 else 0
#     else:
#         ratio = val_salt / val_acid

#     # 顯示條件：
#     # 1. 數值不同 (Diff > 0.001)
#     # 2. 且 (差異比例 > 1.2 倍 或 純粹是從 0 變成有值)
#     if diff > 0.001 and (ratio > 1.2 or ratio < 0.8):
        
#         note = ""
#         if 'Z' in col: note = "Atomic Z (Mg)"
#         elif 'Acid' in col: note = "Acid Count"
#         elif 'Base' in col: note = "Base Count"
#         elif 'PEOE' in col: note = "Charge/Elec"
#         elif 'MW' in col: note = "Mol Weight"
        
#         # 只顯示有意義的標註，或是差異巨大的特徵 (前 30 個就好，不然太多)
#         if count < 30 or note != "":
#              print(f"{col:<25} | {val_acid:<12.4f} | {val_salt:<12.4f} | {ratio:<8.1f} | {note}")
#              count += 1
for col in df.columns:
    # ⚠️ FIX: 強制轉為 float，避免 Boolean 相減報錯
    try:
        val_acid = float(df.loc["Stearic_Acid", col])
        val_salt = float(df.loc["Mg_Stearate", col])
    except:
        continue # 如果真的轉不過去就跳過
    
    # 計算差異
    diff = abs(val_salt - val_acid)
    
    # 避免除以零
    if val_acid == 0:
        ratio = 999.0 if val_salt > 0 else 0
    else:
        ratio = val_salt / val_acid

    # 顯示條件：
    # 1. 數值不同 (Diff > 0.001)
    # 2. 且 (差異比例 > 1.2 倍 或 純粹是從 0 變成有值)
    if diff > 0.001 and (ratio > 1.2 or ratio < 0.8):
        
        note = ""
        if 'Z' in col: note = "Atomic Z (Mg)"
        elif 'Acid' in col: note = "Acid Count"
        elif 'Base' in col: note = "Base Count"
        elif 'PEOE' in col: note = "Charge/Elec"
        elif 'MW' in col: note = "Mol Weight"
        
        # 只顯示有意義的標註，或是差異巨大的特徵 (前 50 個就好)
        if count < 50 or note != "":
             print(f"{col:<25} | {val_acid:<12.4f} | {val_salt:<12.4f} | {ratio:<8.1f} | {note}")
             count += 1

print("="*90)
print(f"📊 Total differing features found: {count}+")


print("="*80)
print(f"📊 Total distinctive features found: {count}")

if found_mg_signal:
    print("✅ SUCCESS: Metal-specific or Atomic Number (Z) features detected.")
    print("   The model can distinguishing the salt from the acid.")
else:
    print("⚠️ WARNING: No direct metal label found, but topological differences exist.")

# ================================================================================
# 🕵️‍♂️  Unique Features detected in Mg Stearate (absent in Stearic Acid)
# ================================================================================
# Feature Name              | Stearic Ac.  | Mg Stearate  | Interpretation
# --------------------------------------------------------------------------------
# ⚙️  Running Feature Hunter v2 (Difference Mode)...

# ==========================================================================================
# 🕵️‍♂️  Top Differences: Stearic Acid vs. Mg Stearate
# ==========================================================================================
# Feature                   | Stearic Ac.  | Mg Stearate  | Diff Ratio | Note
# ------------------------------------------------------------------------------------------
# nAcid                     | 1.0000       | 2.0000       | 2.0      | Acid Count
# SpAbs_A                   | 24.1364      | 0.0000       | 0.0      |
# SpMax_A                   | 1.9932       | 0.0000       | 0.0      |
# SpDiam_A                  | 3.9863       | 0.0000       | 0.0      |
# SpAD_A                    | 24.1364      | 0.0000       | 0.0      |
# SpMAD_A                   | 1.2068       | 0.0000       | 0.0      |
# LogEE_A                   | 3.8136       | 0.0000       | 0.0      |
# VE1_A                     | 4.0844       | 0.0000       | 0.0      |
# VE2_A                     | 0.2042       | 0.0000       | 0.0      |
# VE3_A                     | 2.1003       | 0.0000       | 0.0      |
# VR1_A                     | 125.1633     | 0.0000       | 0.0      |
# VR2_A                     | 6.2582       | 0.0000       | 0.0      |
# VR3_A                     | 5.5228       | 0.0000       | 0.0      |
# nAtom                     | 56.0000      | 111.0000     | 2.0      |
# nHeavyAtom                | 20.0000      | 41.0000      | 2.0      |
# nHetero                   | 2.0000       | 5.0000       | 2.5      |
# nH                        | 36.0000      | 70.0000      | 1.9      |
# nC                        | 18.0000      | 36.0000      | 2.0      |
# nO                        | 2.0000       | 4.0000       | 2.0      |
# ATS0dv                    | 142.0000     | 332.0000     | 2.3      |
# ATS1dv                    | 114.0000     | 244.0000     | 2.1      |
# ATS2dv                    | 118.0000     | 268.0000     | 2.3      |
# ATS3dv                    | 84.0000      | 176.0000     | 2.1      |
# ATS4dv                    | 80.0000      | 168.0000     | 2.1      |
# ATS5dv                    | 76.0000      | 160.0000     | 2.1      |
# ATS6dv                    | 72.0000      | 152.0000     | 2.1      |
# ATS7dv                    | 68.0000      | 144.0000     | 2.1      |
# ATS8dv                    | 64.0000      | 136.0000     | 2.1      |
# ATS0d                     | 112.0000     | 222.0000     | 2.0      |
# ATS1d                     | 142.0000     | 282.0000     | 2.0      |
# ATS2d                     | 225.0000     | 444.0000     | 2.0      |
# ATS3d                     | 263.0000     | 520.0000     | 2.0      |
# ATS4d                     | 248.0000     | 488.0000     | 2.0      | 
# ATS5d                     | 232.0000     | 456.0000     | 2.0      |
# ATS6d                     | 216.0000     | 424.0000     | 2.0      |
# ATS7d                     | 200.0000     | 392.0000     | 2.0      |
# ATS8d                     | 184.0000     | 360.0000     | 2.0      |
# ATS0s                     | 163.7778     | 0.0000       | 0.0      |
# ATS1s                     | 120.9167     | 0.0000       | 0.0      |
# ATS2s                     | 221.0000     | 0.0000       | 0.0      |
# ATS3s                     | 250.5833     | 0.0000       | 0.0      |
# ATS4s                     | 233.3333     | 0.0000       | 0.0      |
# ATS5s                     | 221.0833     | 0.0000       | 0.0      |
# ATS6s                     | 208.8333     | 0.0000       | 0.0      |
# ATS7s                     | 196.5833     | 0.0000       | 0.0      |
# ATS8s                     | 184.3333     | 0.0000       | 0.0      |
# ATS0Z                     | 812.0000     | 1766.0000    | 2.2      | Atomic Z (Mg)
# ATS1Z                     | 926.0000     | 1836.0000    | 2.0      | Atomic Z (Mg)
# ATS2Z                     | 1163.0000    | 2314.0000    | 2.0      | Atomic Z (Mg)
# ATS3Z                     | 1126.0000    | 2224.0000    | 2.0      | Atomic Z (Mg)
# ATS4Z                     | 1056.0000    | 2096.0000    | 2.0      | Atomic Z (Mg)
# ATS5Z                     | 992.0000     | 1968.0000    | 2.0      | Atomic Z (Mg)
# ATS6Z                     | 928.0000     | 1840.0000    | 2.0      | Atomic Z (Mg)
# ATS7Z                     | 864.0000     | 1712.0000    | 2.0      | Atomic Z (Mg)
# ATS8Z                     | 800.0000     | 1584.0000    | 2.0      | Atomic Z (Mg)
# ATSC0Z                    | 354.8571     | 784.9189     | 2.2      | Atomic Z (Mg)
# ATSC1Z                    | -13.5918     | -45.6508     | 3.4      | Atomic Z (Mg)
# ATSC2Z                    | -114.5510    | -247.7385    | 2.2      | Atomic Z (Mg)
# ATSC3Z                    | -13.1837     | 17.7327      | -1.3     | Atomic Z (Mg)
# ATSC4Z                    | 2.9388       | 16.0438      | 5.5      | Atomic Z (Mg)
# ATSC5Z                    | 2.6122       | 14.3550      | 5.5      | Atomic Z (Mg)
# ATSC6Z                    | 2.2857       | 12.6662      | 5.5      | Atomic Z (Mg)
# ATSC7Z                    | 1.9592       | 10.9774      | 5.6      | Atomic Z (Mg)
# ATSC8Z                    | 1.6327       | 9.2885       | 5.7      | Atomic Z (Mg)
# AATSC1Z                   | -0.2471      | -0.4227      | 1.7      | Atomic Z (Mg)
# AATSC3Z                   | -0.0867      | 0.0591       | -0.7     | Atomic Z (Mg)
# AATSC4Z                   | 0.0204       | 0.0569       | 2.8      | Atomic Z (Mg)
# AATSC5Z                   | 0.0193       | 0.0544       | 2.8      | Atomic Z (Mg)
# AATSC6Z                   | 0.0181       | 0.0515       | 2.8      | Atomic Z (Mg)
# AATSC7Z                   | 0.0167       | 0.0481       | 2.9      | Atomic Z (Mg)
# AATSC8Z                   | 0.0151       | 0.0442       | 2.9      | Atomic Z (Mg)
# MATS1Z                    | -0.0390      | -0.0598      | 1.5      | Atomic Z (Mg)
# MATS3Z                    | -0.0137      | 0.0084       | -0.6     | Atomic Z (Mg)
# MATS4Z                    | 0.0032       | 0.0080       | 2.5      | Atomic Z (Mg)
# MATS5Z                    | 0.0031       | 0.0077       | 2.5      | Atomic Z (Mg)
# MATS6Z                    | 0.0029       | 0.0073       | 2.5      | Atomic Z (Mg)
# MATS7Z                    | 0.0026       | 0.0068       | 2.6      | Atomic Z (Mg)
# MATS8Z                    | 0.0024       | 0.0063       | 2.6      | Atomic Z (Mg)
# BCUTZ-1h                  | 8.0286       | 0.0000       | 0.0      | Atomic Z (Mg)
# BCUTZ-1l                  | 5.8035       | 0.0000       | 0.0      | Atomic Z (Mg)
# SpAbs_DzZ                 | 268.5892     | 0.0000       | 0.0      | Atomic Z (Mg)
# SpMax_DzZ                 | 134.5446     | 0.0000       | 0.0      | Atomic Z (Mg)
# SpDiam_DzZ                | 215.0270     | 0.0000       | 0.0      | Atomic Z (Mg)
# SpAD_DzZ                  | 269.0392     | 0.0000       | 0.0      | Atomic Z (Mg)
# SpMAD_DzZ                 | 13.4520      | 0.0000       | 0.0      | Atomic Z (Mg)
# LogEE_DzZ                 | 134.5446     | 0.0000       | 0.0      | Atomic Z (Mg)
# SM1_DzZ                   | 0.5000       | 0.0000       | 0.0      | Atomic Z (Mg)
# VE1_DzZ                   | 4.4026       | 0.0000       | 0.0      | Atomic Z (Mg)
# VE2_DzZ                   | 0.2201       | 0.0000       | 0.0      | Atomic Z (Mg)
# VE3_DzZ                   | 2.1753       | 0.0000       | 0.0      | Atomic Z (Mg)
# VR1_DzZ                   | 90.2120      | 0.0000       | 0.0      | Atomic Z (Mg)
# VR2_DzZ                   | 4.5106       | 0.0000       | 0.0      | Atomic Z (Mg)
# VR3_DzZ                   | 5.1953       | 0.0000       | 0.0      | Atomic Z (Mg)
# SZ                        | 26.6667      | 55.0000      | 2.1      | Atomic Z (Mg)
# ZMIC0                     | 74.3412      | 149.4271     | 2.0      | Atomic Z (Mg)
# ZMIC1                     | 64.2772      | 129.0213     | 2.0      | Atomic Z (Mg)
# ZMIC2                     | 58.5180      | 117.6956     | 2.0      | Atomic Z (Mg)
# ZMIC3                     | 51.6284      | 104.0218     | 2.0      | Atomic Z (Mg)
# ZMIC4                     | 44.2974      | 89.4198      | 2.0      | Atomic Z (Mg)
# ZMIC5                     | 36.9133      | 74.6733      | 2.0      | Atomic Z (Mg)
# PEOE_VSA1                 | 5.1065       | 19.8021      | 3.9      | Charge/Elec
# PEOE_VSA2                 | 4.7945       | 0.0000       | 0.0      | Charge/Elec
# PEOE_VSA6                 | 96.8152      | 193.6305     | 2.0      | Charge/Elec
# PEOE_VSA7                 | 6.4208       | 25.6833      | 4.0      | Charge/Elec
# PEOE_VSA8                 | 6.4208       | 11.9386      | 1.9      | Charge/Elec
# MWC01                     | 19.0000      | 38.0000      | 2.0      | Mol Weight
# TMWC10                    | 102.5198     | 148.7448     | 1.5      | Mol Weight
# MW                        | 284.2715     | 590.5125     | 2.1      | Mol Weight
# Zagreb1                   | 76.0000      | 152.0000     | 2.0      | Atomic Z (Mg)
# Zagreb2                   | 74.0000      | 148.0000     | 2.0      | Atomic Z (Mg)
# mZagreb1                  | 7.1111       | 0.0000       | 0.0      | Atomic Z (Mg)
# mZagreb2                  | 5.0833       | 10.1667      | 2.0      | Atomic Z (Mg)
# ==========================================================================================
# 📊 Total differing features found: 112+
# ================================================================================
# 📊 Total distinctive features found: 112
# ⚠️ WARNING: No direct metal label found, but topological differences exist.