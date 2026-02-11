# import joblib
# import pandas as pd

# # 1. 載入模型與特徵名稱列表
# MODEL_FILE = '0210_modelered_rf_model.pkl'
# FEATURE_FILE = '0210_mordred_features_list.pkl'

# print("📂 Loading model and features...")
# rf_model = joblib.load(MODEL_FILE)
# features = joblib.load(FEATURE_FILE)

# # 2. 提取特徵重要性 (Feature Importances)
# # 這是基於 Gini Impurity Decrease 計算出來的分數
# importances = rf_model.feature_importances_

# # 3. 建立 DataFrame 並排序
# df_importance = pd.DataFrame({
#     'Feature': features,
#     'Importance_Score': importances
# })

# # 依重要性由高到低排序
# df_importance = df_importance.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

# # 4. 加上百分比方便閱讀
# df_importance['Contribution (%)'] = (df_importance['Importance_Score'] * 100).round(2)

# # ==========================================
# # 印出 Top 20 最具資訊量的切分特徵
# # ==========================================
# print("\n" + "="*50)
# print("🏆 Top 20 Most Informative Features (Mordred)")
# print("="*50)
# print(df_importance[['Feature', 'Contribution (%)']].head(20).to_string(index=False))

# # (可選) 匯出成 CSV 給 Eddie 看
# OUTPUT_CSV = "0210_RF_Feature_Importance.csv"
# df_importance.to_csv(OUTPUT_CSV, index=False)
# print(f"\n💾 Full feature importance saved to: {OUTPUT_CSV}")




# # ==================================================
# # 🏆 Top 20 Most Informative Features for API and Excipient Compatibility  (Mordred)
# # ==================================================
# #       Feature  Contribution (%)
# # API_nAromBond              1.40
# #     API_SRW09              1.27
# #    API_naRing              1.21
# #    API_Xch-6d              1.18
# #     API_SRW05              1.14
# #   API_n5aRing              1.12
# #   API_naHRing              1.10
# #   API_n5HRing              0.85
# #     API_piPC2              0.83
# #  API_SMR_VSA3              0.72
# # API_LabuteASA              0.71
# #     API_piPC8              0.70
# #     API_Xp-4d              0.70
# #     API_ATS1m              0.70
# # API_nAromAtom              0.69
# #    API_n5Ring              0.67
# #     API_MWC09              0.67
# #     API_SRW07              0.62
# #  API_nBondsKD              0.60
# #    API_MAXaaN              0.56


import joblib
import pandas as pd

# 1. 載入模型與特徵名稱列表
MODEL_FILE = '0210_modelered_rf_model.pkl'
FEATURE_FILE = '0210_mordred_features_list.pkl'

print("📂 Loading model and features...")
rf_model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

# 2. 建立完整的 DataFrame
df_importance = pd.DataFrame({
    'Feature': features,
    'Importance_Score': rf_model.feature_importances_
})

# 3. 🎯 關鍵步驟：只保留 "EXP_" (賦形劑) 開頭的特徵
df_exp = df_importance[df_importance['Feature'].str.startswith('EXP_')].copy()

# 4. 排序並計算貢獻度
df_exp = df_exp.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

# 算兩種百分比給您看：
# 1. Global_Contribution: 在所有(含API)特徵中佔了多少 %
# 2. Relative_Contribution: 在所有「賦形劑特徵」自己內部佔了多少 % (看相對重要性)
total_exp_score = df_exp['Importance_Score'].sum()

df_exp['Global (%)'] = (df_exp['Importance_Score'] * 100).round(3)
df_exp['Relative_in_EXP (%)'] = ((df_exp['Importance_Score'] / total_exp_score) * 100).round(2)

# 5. 印出 Top 20 賦形劑特徵
print("\n" + "="*65)
print("🧪 Top 20 Most Informative EXCIPIENT Features (EXP_)")
print("="*65)
print(df_exp[['Feature', 'Global (%)', 'Relative_in_EXP (%)']].head(20).to_string(index=False))

# 6. 匯出成 CSV
OUTPUT_CSV = "0211_RF_Excipient_Importance.csv"
df_exp.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Excipient feature importance saved to: {OUTPUT_CSV}")