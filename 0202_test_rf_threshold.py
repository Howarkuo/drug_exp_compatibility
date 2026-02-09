import joblib
import numpy as np

# 1. Load your model
model = joblib.load('0201_my_rf_model.pkl')

# 2. Create some dummy data (or use real vectors) just to check logic
#    (Random 200-dim vector representing an API+Excipient pair)
dummy_X = np.random.rand(5, 200) 

# 3. Get both Probability and Prediction
probs = model.predict_proba(dummy_X)[:, 1]  # Probability of Class 1 (Incompatible)
preds = model.predict(dummy_X)              # Final decision (0 or 1)

print(f"{'Probability':<15} | {'Prediction':<10} | {'Is Thresh 0.5?'}")
print("-" * 45)

for p, label in zip(probs, preds):
    # Check if the logic holds
    logic_check = (p > 0.5 and label == 1) or (p <= 0.5 and label == 0)
    print(f"{p*100:.1f}%          | {label:<10} | {logic_check}")

# PS C:\Users\howar\Desktop\drug_exp_compatibility> poetry run python .\0202_test_rf_threshold.py
# Probability     | Prediction | Is Thresh 0.5?
# ---------------------------------------------
# 97.0%          | 1          | True
# 91.0%          | 1          | True
# 91.5%          | 1          | True
# 94.0%          | 1          | True
# 91.0%          | 1          | True