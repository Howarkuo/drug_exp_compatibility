import pandas as pd
import os

# 設定檔名 (請確認您的檔案是 .csv 還是 .xlsx)
# 如果是 GitHub 下載的原始檔通常是 dataset.csv
INPUT_FILE = '2024_Drug_compatibility_dataset.xlsx' 


def check_imbalance():
    if not os.path.exists(INPUT_FILE):
        print(f" Error: '{INPUT_FILE}' not found.")
        return

    # 嘗試讀取 (自動偵測 csv 或 excel)
    try:
        if INPUT_FILE.endswith('.csv'):
            df = pd.read_csv(INPUT_FILE)
        else:
            df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f" Error reading file: {e}")
        return

    print(f"--- Data Analysis for {INPUT_FILE} ---")
    print(f"Total rows: {len(df)}")
    
    # 🎯 關鍵修改：設定成您截圖中精確的欄位名稱
    target_col = 'Outcome (1: incompatible; 0 compatible)'

    if target_col in df.columns:
        counts = df[target_col].value_counts()
        print("\nClass Distribution:")
        print(counts)
        
        # 0 = compatible, 1 = incompatible
        # 通常 0 (Compatible) 會比較多
        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)
        
        if count_1 > 0:
            ratio = count_0 / count_1
            print(f"\nImbalance Ratio (Compatible : Incompatible) = {ratio:.2f} : 1")
            
            if ratio > 5:
                print(" CONFIRMED: High class imbalance detected!")
                print("   (Eddie was right. We likely need SMOTE or Class Weights)")
            else:
                print(" Data is relatively balanced.")
        else:
            print("Warning: No 'Incompatible' (1) data found!")
    else:
        print(f"\n Column '{target_col}' not found!")
        print("Columns detected:", df.columns.tolist())

if __name__ == "__main__":
    check_imbalance()


#     --- Data Analysis for 2024_Drug_compatibility_dataset.xlsx ---
# Total rows: 3544

# Class Distribution:
# Outcome (1: incompatible; 0 compatible)
# 0    3200
# 1     344
# Name: count, dtype: int64