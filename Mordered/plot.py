import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. Input Data (Hardcoded Data)
# ==========================================
# Data from your logs: [[TN, FP], [FN, TP]]
cm = np.array([[622, 18], 
                [19, 50]])

# Data from your report to be displayed in the title
accuracy = 0.9478
recall = 0.72  # Class 1 Recall
precision = 0.74

# ==========================================
# 2. Configure Chart Style
# ==========================================
plt.figure(figsize=(8, 6))
sns.set_context("notebook", font_scale=1.2)

# Define readable labels
labels = ['Compatible (0)', 'Incompatible (1)']

# ==========================================
# 3. Plot Heatmap
# ==========================================
# fmt='d': Displays as integers
# cmap='Blues': Blue color palette
# annot=True: Displays numbers inside the boxes

ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels,
                 annot_kws={"size": 18, "weight": "bold"},
                 cbar=False) # Hide color bar for a cleaner look

# ==========================================
# 4. Add Title and Labels
# ==========================================
plt.title(f'Confusion Matrix: MLP + Mordred\n(Accuracy: {accuracy:.2%}, Recall: {recall:.2f})', 
          fontsize=16, fontweight='bold', pad=20)

plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')

# Fine-tune layout
plt.tight_layout()

# ==========================================
# 5. Output and Save
# ==========================================
output_file = 'confusion_matrix_final_mlp.png'
plt.savefig(output_file, dpi=300)
print(f"✅ Chart saved as: {output_file}")

# Display text-based classification report (Reconstructed)
print("\n" + "="*55)
print("📄 Classification Report (Reconstructed)")
print("="*55)
print(f"{'':<12} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
print(f"{'0':<12} {0.97:<10.2f} {0.97:<10.2f} {0.97:<10.2f} {640:<10}")
print(f"{'1':<12} {precision:<10.2f} {recall:<10.2f} {0.73:<10.2f} {69:<10}")
print("-" * 55)
print(f"Accuracy: {accuracy:.4f}")
print("="*55)

plt.show()