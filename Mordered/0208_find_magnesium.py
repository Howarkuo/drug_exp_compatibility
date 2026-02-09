import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# 1. Define Molecule
molecules = {
    "Stearic Acid": "CCCCCCCCCCCCCCCCCC(=O)O",
    "Mg_Stearate": "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]"
}

print("⚙️ Running Feature Hunter...")

# 2. Count all properties
mols = [Chem.MolFromSmiles(smi) for smi in molecules.values()]
calc = Calculator(descriptors, ignore_3D=True)
df = calc.pandas(mols, nproc=1, quiet=True)

# Make up 0s
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df.index = molecules.keys()

# 3. find Mg_Stearate > 0 but Vitamin_C == 0 
print("\n" + "="*60)
print("🕵️‍♂️ 差異特徵列表 (Mg Stearate Unique Features)")
print("="*60)

found_mg = False
count = 0

for col in df.columns:
    val_vit = df.loc["Stearic Acid", col]
    val_mg = df.loc["Mg_Stearate", col]
    
    # 條件：Vit C 沒有，但 Mg Stearate 有
    if val_vit == 0 and val_mg > 0:
        # 過濾掉太通用的特徵，專注找可能是金屬或特定結構的
        print(f"{col:<20} | Vit C: {val_vit:<5} | Mg St: {val_mg:<5}")
        count += 1
        
        # 檢查名字裡有沒有 Z (原子序) 或 Metal
        if 'Z' in col or 'Metal' in col or 'Mg' in col:
            found_mg = True

print("="*60)
print(f"共找到 {count} 個 Mg Stearate 獨有的特徵。")

if found_mg:
    print("✅ 找到了金屬/原子序相關特徵！模型將能區分它們。")
else:
    print("⚠️ 沒看到明顯的 Metal 標籤，但上述特徵足夠讓 MLP 區分兩者。")




# ============================================================
# 🕵️‍♂️ 差異特徵列表 (Mg Stearate Unique Features)
# ============================================================
# nAcid                | Vit C: 0     | Mg St: 2
# ATS7dv               | Vit C: 0.0   | Mg St: 144.0
# ATS8dv               | Vit C: 0.0   | Mg St: 136.0
# AATS7dv              | Vit C: 0.0   | Mg St: 0.631578947368421
# AATS8dv              | Vit C: 0.0   | Mg St: 0.6476190476190476
# GATS8dv              | Vit C: 0.0   | Mg St: 0.8791151497095191
# GATS7d               | Vit C: 0.0   | Mg St: 0.8855666677216194
# GATS8d               | Vit C: 0.0   | Mg St: 0.8915471179451939
# GATS8Z               | Vit C: 0.0   | Mg St: 0.8535306829845769
# GATS8m               | Vit C: 0.0   | Mg St: 0.8618078143086656
# GATS8v               | Vit C: 0.0   | Mg St: 0.954733832478533
# GATS8se              | Vit C: 0.0   | Mg St: 0.593726504121097
# GATS8pe              | Vit C: 0.0   | Mg St: 0.7696492917883111
# GATS8are             | Vit C: 0.0   | Mg St: 0.7428581833537017
# GATS8p               | Vit C: 0.0   | Mg St: 0.2035736884627803
# GATS8i               | Vit C: 0.0   | Mg St: 0.8101020007854209
# NsCH3                | Vit C: 0     | Mg St: 2
# SsCH3                | Vit C: 0.0   | Mg St: 4.534462692176307
# MAXsCH3              | Vit C: 0.0   | Mg St: 2.2672313460881535
# MINsCH3              | Vit C: 0.0   | Mg St: 2.2672313460881535
# PEOE_VSA6            | Vit C: 0.0   | Mg St: 193.63047984130958
# PEOE_VSA7            | Vit C: 0.0   | Mg St: 25.683286491704038
# PEOE_VSA8            | Vit C: 0.0   | Mg St: 11.938610575903699
# SlogP_VSA1           | Vit C: 0.0   | Mg St: 10.213054789681411
# SlogP_VSA5           | Vit C: 0.0   | Mg St: 219.31376633301363
# EState_VSA2          | Vit C: 0.0   | Mg St: 35.894434948091806
# EState_VSA4          | Vit C: 0.0   | Mg St: 25.683286491704038
# EState_VSA5          | Vit C: 0.0   | Mg St: 166.9413621960763
# MDEC-11              | Vit C: 0.0   | Mg St: 1e-08
# MDEC-12              | Vit C: 0.0   | Mg St: 0.0024542037291359147
# MDEC-13              | Vit C: 0.0   | Mg St: 9.70142500145332e-05
# MPC9                 | Vit C: 0     | Mg St: 22
# MPC10                | Vit C: 0     | Mg St: 20
# piPC9                | Vit C: 0.0   | Mg St: 3.2188758248682006
# piPC10               | Vit C: 0.0   | Mg St: 3.1354942159291497
# GGI6                 | Vit C: 0.0   | Mg St: 0.16326530612244897
# GGI7                 | Vit C: 0.0   | Mg St: 0.125
# GGI8                 | Vit C: 0.0   | Mg St: 0.0987654320987654
# GGI9                 | Vit C: 0.0   | Mg St: 0.08000000000000002
# GGI10                | Vit C: 0.0   | Mg St: 0.06611570247933884
# JGI6                 | Vit C: 0.0   | Mg St: 0.0058309037900874635
# JGI7                 | Vit C: 0.0   | Mg St: 0.004807692307692308
# JGI8                 | Vit C: 0.0   | Mg St: 0.004115226337448558
# JGI9                 | Vit C: 0.0   | Mg St: 0.003636363636363638
# JGI10                | Vit C: 0.0   | Mg St: 0.0033057851239669425
# ============================================================
# 共找到 45 個 Mg Stearate 獨有的特徵。
# ✅ 找到了金屬/原子序相關特徵！模型將能區分它們。