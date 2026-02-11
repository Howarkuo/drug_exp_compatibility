# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from mordred import Calculator, descriptors
# from sklearn.ensemble import RandomForestClassifier
# from tqdm import tqdm
# import multiprocessing

# # On Windows, all executable code MUST be inside this 'if' block
# if __name__ == "__main__":
#     # ---------------------------------------------------------
#     # 1. CREATE DUMMY DATA
#     # ---------------------------------------------------------
#     print("🧪 Creating dummy data...")
#     # Repeating the list just to make the progress bar visible
#     base_smiles = [
#         "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
#         "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
#         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
#         "CC(=O)NC1=CC=C(C=C1)O", # Paracetamol
#         "CN(C)C(=N)NC(=N)NC1=CC=CC=C1" # Metformin
#     ]
#     # Multiply by 100 so we have 500 molecules (enough to see the bar move)
#     smiles_data = base_smiles * 100 
#     labels = [0, 1, 1, 0, 1] * 100

#     # ---------------------------------------------------------
#     # 2. SETUP MORDRED
#     # ---------------------------------------------------------
#     print("⚙️ Setting up Mordred...")
#     calc = Calculator(descriptors, ignore_3D=True)

#     # ---------------------------------------------------------
#     # 3. CONVERT SMILES -> RDKIT MOLS (With TQDM)
#     # ---------------------------------------------------------
#     print(f"\n🔹 Step 1: Converting {len(smiles_data)} SMILES to molecules...")
#     mols = []
#     # Using tqdm here so you see a progress bar for conversion
#     for smi in tqdm(smiles_data, desc="Converting"):
#         mols.append(Chem.MolFromSmiles(smi))

#     # ---------------------------------------------------------
#     # 4. CALCULATE DESCRIPTORS (With Built-in Bar)
#     # ---------------------------------------------------------
#     print("\n🔹 Step 2: Calculating features...")
    
#     # quiet=False forces Mordred to show its own progress bar
#     # n_proc=1 is SAFEST for Windows if you still have issues, 
#     # but try default first.
#     try:
#         df_features = calc.pandas(mols, quiet=False)
#     except RuntimeError:
#         print("⚠️ Multiprocessing error caught! Retrying with n_proc=1...")
#         df_features = calc.pandas(mols, n_proc=1, quiet=False)

#     # ---------------------------------------------------------
#     # 5. CLEAN & TRAIN
#     # ---------------------------------------------------------
#     # Convert to numeric, turn errors to 0
#     df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)

#     print("\n🤖 Training Test RF Model...")
#     rf = RandomForestClassifier(n_estimators=10, random_state=42)
#     rf.fit(df_features, labels)

#     print(f"🎉 Success! Model trained on {df_features.shape} matrix.")
#     print("   (You are now ready to run the main script!)")

# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from mordred import Calculator, descriptors
# from sklearn.ensemble import RandomForestClassifier
# from tqdm import tqdm
# import multiprocessing

# # 🛑 ON WINDOWS, THIS BLOCK IS MANDATORY
# if __name__ == "__main__":
    
#     # 1. SETUP MORDRED
#     print("⚙️ Setting up Mordred...")
#     calc = Calculator(descriptors, ignore_3D=True)

#     # ✅ PRINT COUNT HERE (Before doing any work)
#     print(f"📊 Plan: Mordred is configured to calculate {len(calc.descriptors)} descriptors.")

#     # 2. CREATE DUMMY DATA
#     print("\n🧪 Creating dummy data...")
#     # 5 molecules repeated 20 times = 100 molecules
#     base_smiles = [
#         "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
#         "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
#         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
#         "CC(=O)NC1=CC=C(C=C1)O", # Paracetamol
#         "CN(C)C(=N)NC(=N)NC1=CC=CC=C1" # Metformin
#     ]
#     smiles_data = base_smiles * 20 
#     labels = [0, 1, 1, 0, 1] * 20

#     # 3. CONVERT SMILES (With Progress Bar)
#     print(f"\n🔹 Step 1: Converting {len(smiles_data)} SMILES to molecules...")
#     mols = []
#     for smi in tqdm(smiles_data, desc="Converting"):
#         mols.append(Chem.MolFromSmiles(smi))

#     # 4. CALCULATE (With Progress Bar)
#     print("\n🔹 Step 2: Calculating features...")
    
#     # Use n_proc=1 to be safe on Windows if you had crashes before
#     df_features = calc.pandas(mols, quiet=False, nproc=1)

#     # 5. CLEAN DATA
#     df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # ✅ PRINT FINAL COUNT HERE
#     print(f"✅ Result: Successfully generated {df_features.shape[1]} features.")

#     # 6. TRAIN TINY MODEL
#     print("\n🤖 Training Test RF Model...")
#     rf = RandomForestClassifier(n_estimators=10, random_state=42)
#     rf.fit(df_features, labels)
    
#     print("🎉 Success! The test is complete.")

#     # 1613 Features


# import pandas as pd
# import numpy as np
# import os
# import joblib
# from rdkit import Chem
# from mordred import Calculator, descriptors
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.feature_selection import VarianceThreshold
# from imblearn.over_sampling import SMOTE
# from tqdm import tqdm

# # ==============================================================================
# # 1. SETUP & HELPER FUNCTIONS
# # ==============================================================================

# def generate_mordred_features(smiles_list, prefix):
#     """
#     Generates 2D molecular descriptors using Mordred.
#     Handles errors and invalid SMILES automatically.
#     """
#     print(f"\n⚙️  Generating {prefix} features using Mordred...")
    
#     # Initialize Calculator (Calculates ~1613 2D descriptors)
#     # ignore_3D=True ensures we don't need 3D coordinates (which require slow embedding)
#     calc = Calculator(descriptors, ignore_3D=True)
    
#     # --- Step A: Convert SMILES to RDKit Molecules ---
#     mols = []
#     valid_indices = [] # Keep track of which rows are valid
    
#     print(f"   - Step 1: Converting {len(smiles_list)} SMILES strings to molecules...")
#     for i, smi in enumerate(tqdm(smiles_list, desc="   - Converting")):
#         try:
#             mol = Chem.MolFromSmiles(str(smi))
#             # Check if molecule is valid and has atoms
#             if mol and mol.GetNumAtoms() > 0:
#                 mols.append(mol)
#                 valid_indices.append(i)
#             else:
#                 mols.append(None)
#         except:
#             mols.append(None)

#     # Filter out None values for the calculation step
#     valid_mols = [m for m in mols if m is not None]
    
#     if not valid_mols:
#         print(f"❌ CRITICAL ERROR: No valid molecules found for {prefix}. Check your SMILES data.")
#         return pd.DataFrame()

#     # --- Step B: Calculate Descriptors ---
#     print(f"   - Step 2: Calculating {len(calc.descriptors)} descriptors for {len(valid_mols)} molecules...")
    
#     # nproc=1 is CRITICAL for Windows to prevent "spawn" errors and crashes.
#     # It is still very fast (approx 30-50 mols/sec).
#     try:
#         df_features = calc.pandas(valid_mols, nproc=1, quiet=False)
#     except Exception as e:
#         print(f"❌ Calculation failed: {e}")
#         return pd.DataFrame()

#     # --- Step C: Cleanup & formatting ---
#     # Convert all columns to numeric, turning errors (strings/objects) into NaN
#     df_features = df_features.apply(pd.to_numeric, errors='coerce')
    
#     # Fill NaN and Infinity values with 0
#     df_features = df_features.fillna(0)
    
#     # Create a DataFrame that matches the ORIGINAL length of the input list
#     # (This ensures we don't lose row alignment if some SMILES were invalid)
#     final_df = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_features.columns)
#     final_df.iloc[valid_indices] = df_features.values
    
#     # Add prefix to column names (e.g. "API_MW", "EXP_MW") to avoid collisions
#     final_df = final_df.add_prefix(prefix)
    
#     print(f"✅ Success! Generated {final_df.shape[1]} features for {prefix}.")
#     return final_df

# # ==============================================================================
# # 2. MAIN EXECUTION BLOCK (Mandatory for Windows Multiprocessing)
# # ==============================================================================
# if __name__ == "__main__":
    
#     # --- Configuration ---
#     FILE_PATH = "2024_Drug_compatibility_dataset.xlsx" # <--- Your Excel File
#     API_COL = "API_Smiles"          # Check your Excel file for exact column name
#     EXP_COL = "Excipient_Smiles"    # Check your Excel file for exact column name
#     LABEL_COL = "Outcome (1: incompatible; 0 compatible)"    
#     # --- 1. Load Data ---
#     if not os.path.exists(FILE_PATH):
#         print(f"❌ Error: File '{FILE_PATH}' not found.")
#         exit()
        
#     print(f"📂 Loading dataset: {FILE_PATH}")
#     try:
#         df = pd.read_excel(FILE_PATH)
#     except Exception as e:
#         print(f"❌ Error reading Excel file: {e}")
#         exit()

#     # Verify columns exist
#     missing_cols = [c for c in [API_COL, EXP_COL, LABEL_COL] if c not in df.columns]
#     if missing_cols:
#         print(f"❌ Missing columns in Excel file: {missing_cols}")
#         print(f"   Found columns: {list(df.columns)}")
#         print("   Please rename the columns in your code or Excel file.")
#         exit()

#     print(f"   loaded {len(df)} rows.")

#     # --- 2. Generate Features (API + Excipient) ---
#     df_api = generate_mordred_features(df[API_COL].tolist(), prefix="API_")
#     df_exp = generate_mordred_features(df[EXP_COL].tolist(), prefix="EXP_")
    
#     if df_api.empty or df_exp.empty:
#         print("❌ Feature generation failed. Aborting.")
#         exit()

#     # --- 3. Combine & Clean Data ---
#     print("\n🔗 Combining API and Excipient features...")
#     X = pd.concat([df_api, df_exp], axis=1)
#     y = df[LABEL_COL]

#     print(f"   Raw Feature Shape: {X.shape}")
    
#     # Remove "Constant" features (Columns that are all 0s or all same value)
#     # This drastically speeds up training and removes noise.
#     print("🧹 Cleaning data (Removing constant features)...")
#     selector = VarianceThreshold(threshold=0)
#     X_clean = selector.fit_transform(X)
    
#     # Get the names of the kept features
#     kept_features = X.columns[selector.get_support()]
#     X = pd.DataFrame(X_clean, columns=kept_features)
    
#     print(f"   Cleaned Feature Shape: {X.shape} (Removed {df_api.shape[1]*2 - X.shape[1]} useless columns)")

#     # --- 4. Balance Data (SMOTE) ---
#     print("\n⚖️  Applying SMOTE to balance classes...")
#     try:
#         smote = SMOTE(random_state=42)
#         X_resampled, y_resampled = smote.fit_resample(X, y)
#         print(f"   Original class split: {y.value_counts().to_dict()}")
#         print(f"   Balanced class split: {y_resampled.value_counts().to_dict()}")
#     except ValueError as e:
#         print(f"⚠️ SMOTE failed (dataset might be too small). Using original data. Error: {e}")
#         X_resampled, y_resampled = X, y

#     # --- 5. Train Model (Random Forest) ---
#     print("\n🤖 Training Random Forest Classifier...")
    
#     # Initialize RF
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
#     # 5-Fold Cross Validation
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(rf, X_resampled, y_resampled, cv=cv, scoring='accuracy')
    
#     print(f"\n🏆 Model Performance (5-Fold CV):")
#     print(f"   Mean Accuracy: {scores.mean():.4f}")
#     print(f"   Std Deviation: {scores.std():.4f}")

#     # --- 6. Save Final Model ---
#     print("\n💾 Retraining on full data and saving...")
#     rf.fit(X_resampled, y_resampled)
    
#     model_filename = "0208_modelered_rf_model.pkl"
#     joblib.dump(rf, model_filename)
#     print(f"✅ Model saved successfully as: {model_filename}")
#     print("🎉 Done!")

    


#     🧪 COMPATIBILITY PREDICTION (Mordred Model)

# 💊 API: Vitamin C (Ascorbic Acid)

# ======================================================================

# Excipient            | Prediction      | Probability

# ----------------------------------------------------------------------



# Mg Stearate          | 🟢 Compatible    | 50.00%

# Fluorinated Amide    | 🔴 Incompatible  | 57.00%

# Cellulose            | 🔴 Incompatible  | 52.00%

# Stearic Acid         | 🔴 Incompatible  | 52.00%

# Mannitol             | 🟢 Compatible    | 53.00%

# Silicon Dioxide      | 🟢 Compatible    | 63.00%



import pandas as pd
import numpy as np
import os
import joblib
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. HELPER FUNCTION
# ==========================================
def generate_mordred_features(smiles_list, prefix):
    print(f"\n⚙️ Generating {prefix} features using Mordred...")
    
    # 1. Convert to Molecules
    mols = []
    for s in tqdm(smiles_list, desc="   Converting"):
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol and mol.GetNumAtoms() > 0:
                mols.append(mol)
            else:
                mols.append(None)
        except:
            mols.append(None)

    # 2. Filter Valid Molecules
    valid_mols = [m for m in mols if m is not None]
    if not valid_mols:
        print(f"❌ CRITICAL ERROR: No valid molecules found for {prefix}.")
        return pd.DataFrame()

    # 3. Calculate Descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    try:
        # nproc=1 is critical for Windows
        print(f"   Calculating descriptors...")
        df_features = calc.pandas(valid_mols, nproc=1, quiet=True)
    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return pd.DataFrame()

    # 4. Cleanup & Re-align
    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Create full-length dataframe to match original input
    final_df = pd.DataFrame(0, index=range(len(smiles_list)), columns=df_features.columns)
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    final_df.iloc[valid_indices] = df_features.values
    
    return final_df.add_prefix(prefix)

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # --- Configuration ---
    # Check if we need "../" depending on where you run this
    FILE_PATH = "../2024_Drug_compatibility_dataset.xlsx" 
    
    # Output Files
    MODEL_FILE = "0210_modelered_rf_model.pkl"
    FEATURES_FILE = "0210_mordred_features_list.pkl" # ✅ Crucial for prediction

    # --- 1. Load Data ---
    if not os.path.exists(FILE_PATH):
        # Fallback to current folder if ../ fails
        FILE_PATH = "2024_Drug_compatibility_dataset.xlsx"
        if not os.path.exists(FILE_PATH):
            print(f"❌ Error: File not found."); exit()

    print(f"📂 Loading dataset: {FILE_PATH}")
    df = pd.read_excel(FILE_PATH)
    print(f"   Loaded {len(df)} rows.")

    # --- 2. Generate Features ---
    df_api = generate_mordred_features(df['API_Smiles'].tolist(), prefix="API_")
    df_exp = generate_mordred_features(df['Excipient_Smiles'].tolist(), prefix="EXP_")
    
    if df_api.empty or df_exp.empty: exit()

    # --- 3. Combine & Clean (Feature Selection) ---
    print("\n🔗 Combining features...")
    X = pd.concat([df_api, df_exp], axis=1)
    y = df["Outcome (1: incompatible; 0 compatible)"]

    print("🧹 Removing constant features...")
    selector = VarianceThreshold(threshold=0)
    X_clean = selector.fit_transform(X)
    
    # ✅ Capture the kept feature names HERE
    kept_features = X.columns[selector.get_support()]
    X = pd.DataFrame(X_clean, columns=kept_features)
    
    print(f"   Original Features: {df_api.shape[1]*2}")
    print(f"   Final Features:    {X.shape[1]}")

    # --- 4. SMOTE Balancing ---
    print("\n⚖️ Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except:
        print("⚠️ SMOTE failed. Using unbalanced data.")
        X_res, y_res = X, y

    # --- 5. Train Model ---
    print("\n🤖 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # CV Check
    scores = cross_val_score(rf, X_res, y_res, cv=5, scoring='accuracy')
    print(f"   Mean CV Accuracy: {scores.mean():.4f}")

    # Final Fit
    rf.fit(X_res, y_res)
    
    # --- 6. Save Model AND Feature List ---
    print("\n💾 Saving Model & Feature List...")
    joblib.dump(rf, MODEL_FILE)
    joblib.dump(list(kept_features), FEATURES_FILE) # ✅ Saves the list needed for prediction
    
    print(f"✅ Model saved to: {MODEL_FILE}")
    print(f"✅ Feature list saved to: {FEATURES_FILE}")
    print("🎉 Done!")