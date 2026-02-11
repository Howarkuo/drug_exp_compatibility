import pandas as pd
import pubchempy as pcp
import cirpy
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# 關閉 RDKit 警告，保持輸出乾淨
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. 設定
# ==========================================
INPUT_FILE = '../0210_Compatibility_Testset_162.xlsx'
OUTPUT_FILE = '0210_failed_items_report.csv'

# ==========================================
# 2. 讀取資料
# ==========================================
print(f"📂 正在讀取: {INPUT_FILE}...")
try:
    df = pd.read_excel(INPUT_FILE)
    # 簡單更名 Label
    if 'Outcome (1: incompatible; 0 compatible)' in df.columns:
        df = df.rename(columns={'Outcome (1: incompatible; 0 compatible)': 'Label'})
    print(f"📊 總筆數: {len(df)}")
except Exception as e:
    print(f"❌ 讀取失敗: {e}")
    exit()

# ==========================================
# 3. 定義檢測函數
# ==========================================
def check_molecule(cid, role):
    """
    回傳 (SMILES, 錯誤訊息)
    如果有錯誤訊息，表示該分子有問題
    """
    cid = int(cid) if pd.notna(cid) else None
    if not cid:
        return None, "CID Missing"

    # 1. 抓取 SMILES
    smi = None
    try:
        # 嘗試 PubChem
        props = pcp.get_properties('IsomericSMILES', cid)
        if props and 'IsomericSMILES' in props[0]:
            smi = props[0]['IsomericSMILES']
    except:
        pass

    # 嘗試 CIRpy (如果 PubChem 失敗)
    if not smi:
        try:
            c = pcp.Compound.from_cid(cid)
            if c.inchikey:
                smi = cirpy.resolve(c.inchikey, 'smiles')
        except:
            pass

    if not smi:
        return None, f"No SMILES found for CID {cid}"

    # 2. 檢查 RDKit 是否能解析
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, f"RDKit Sanitize Failed ({role})"
    
    try:
        Chem.SanitizeMol(mol)
    except:
        return smi, f"RDKit Sanitize Error ({role})"

    return smi, None # ✅ 成功

# ==========================================
# 4. 開始診斷
# ==========================================
print("🔍 開始診斷每一筆資料...")
failed_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    api_cid = row.get('API_CID')
    exp_cid = row.get('Excipient_CID')
    
    # 檢查 API
    api_smi, api_err = check_molecule(api_cid, "API")
    
    # 檢查 Excipient
    exp_smi, exp_err = check_molecule(exp_cid, "Excipient")
    
    # 判定是否失敗
    if api_err or exp_err:
        failure_reason = []
        if api_err: failure_reason.append(f"[API] {api_err}")
        if exp_err: failure_reason.append(f"[Excipient] {exp_err}")
        
        failed_rows.append({
            'Row_Index': idx + 2, # 對應 Excel 行號 (Header=1)
            'API_CID': api_cid,
            'Excipient_CID': exp_cid,
            'Label': row.get('Label'),
            'API_SMILES': api_smi,
            'Excipient_SMILES': exp_smi,
            'Failure_Reason': "; ".join(failure_reason)
        })

# ==========================================
# 5. 輸出報告
# ==========================================
if failed_rows:
    df_failed = pd.DataFrame(failed_rows)
    df_failed.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig 防止 Excel 亂碼
    
    print("\n" + "="*60)
    print(f"❌ 發現 {len(failed_rows)} 筆失敗資料！")
    print("="*60)
    print(df_failed[['API_CID', 'Excipient_CID', 'Failure_Reason']])
    print(f"\n💾 詳細報告已儲存至: {OUTPUT_FILE}")
    print("(您可以直接把這個 CSV 傳給 Eddie)")
else:
    print("\n✅ 太神奇了！所有資料都成功轉換，沒有發現失敗項目。")