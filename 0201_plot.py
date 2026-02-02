import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Validation Set: [[TN, FP], [FN, TP]]
cm_val = np.array([[627, 13], 
                   [ 23, 46]])

# Test Set: [[TN, FP], [FN, TP]]
cm_test = np.array([[622, 18], 
                    [ 19, 50]])

labels = ['Compatible', 'Incompatible']

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    
    # fmt='d' 表示整數格式, cmap='Blues' 使用藍色系
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"✅ 圖表已儲存為: {filename}")
    plt.show()

plot_confusion_matrix(cm_val, 'Confusion Matrix (Validation Set)', 'confusion_matrix_val.png')

plot_confusion_matrix(cm_test, 'Confusion Matrix (Test Set)', 'confusion_matrix_test.png')