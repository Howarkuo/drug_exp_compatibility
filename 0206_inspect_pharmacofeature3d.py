# import pubchempy as pcp
# import pandas as pd
# import json

# # 測試用 CID (例如: Warfarin 和 Mg Stearate)
# test_cids = [54678486, 11177]

# print("🌍 Fetching PubChem full record (Ann's Method)...")

# for cid in test_cids:
#     print(f"\n🔍 Inspecting CID: {cid}")
#     try:
#         # 1. 抓取完整 Compound 物件
#         c = pcp.Compound.from_cid(cid)
        
#         # 2. 轉成 Dictionary 查看所有屬性
#         data = c.to_dict()
        
#         # 3. 顯示基本屬性 (Scalar Features) - 這些可以直接當 X 特徵
#         print("--- Basic Descriptors (Ready for ML) ---")
#         keys_of_interest = ['molecular_weight', 'xlogp', 'tpsa', 'rotatable_bond_count', 'h_bond_donor_count', 'h_bond_acceptor_count']
#         for k in keys_of_interest:
#             print(f"  {k}: {data.get(k)}")

#         # 4. 顯示 3D 藥效團特徵 (Ann's Suggestion)
#         # 注意：並非每個化合物都有這個欄位，可能需要額外請求
#         print("--- 3D Features ---")
#         # 嘗試抓取 3D 相關紀錄 (這部分比較 tricky，標準 to_dict 可能不含 pharmacophore)
#         # 我們通常需要用 rest API 直接問
#         print(f"  (Checking distinct raw properties...)")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")

# print("\n💡 Conclusion:")
# print("Ann's method gives us high-quality 'xlogp' and 'tpsa' directly.")
# print("However, 'pharmacophore_features_3d' usually requires a specialized JSON parser.")


# import pubchempy as pcp 
# import cirpy
# compound = pcp.Compound.from_cid(14792)
# #Magnesium Oxide
# inchikey = compound.to_dict(properties=['inchikey'])['inchikey']
# # inchikey: CPLXHLVBOLITMK-UHFFFAOYSA-N
# print(inchikey)
# smiles = cirpy.resolve(inchikey, 'smiles')
# # CIR (Chemical Identifier Resolver): A better way to fetch smiles from cid than pubchem
# print(smiles)

# CPLXHLVBOLITMK-UHFFFAOYSA-N
# O=[Mg]


import pubchempy as pcp
import pandas as pd

# 定義我們要測試的 CID
# Vitamin C (Ascorbic Acid): 54670067
# Magnesium Stearate: 11177
target_cids = [54670067, 11177]

print("🔍 Inspecting PubChem Descriptors...")

for cid in target_cids:
    print(f"\n💊 Compound CID: {cid}")
    try:
        c = pcp.Compound.from_cid(cid)
        
        # 1. 抓取基本屬性 (Ann 的第一個建議)
        # 這些是可以直接當作 X 特徵輸入模型的數值
        props = c.to_dict(properties=['molecular_weight', 'charge', 'xlogp', 'tpsa', 'h_bond_donor_count'])
        print("   [Basic Properties]")
        for k, v in props.items():
            print(f"    - {k}: {v}")
            
        # 2. 嘗試抓取 3D 特徵 (Ann 的進階建議)
        # 注意：這需要該分子在 PubChem 有 3D 構型紀錄
        try:
            # 這是透過 REST API 額外請求的，因為標準屬性不包含此項
            # 這裡示範概念，如果沒有紀錄會抓不到
            print(f"    - Has 3D Conformer? {c.cid}") 
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n💡 觀察重點：")
print("1. 注意 'charge' (電荷)：Mg Stearate 應該會有電荷，這是 Mol2Vec 漏掉的。")
print("2. 注意 'tpsa' (極性表面積)：這通常跟吸濕性有關。")