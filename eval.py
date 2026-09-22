import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ================= 配置 =================
WORK_DIR = './model_output'
RESULT_FILE = os.path.join(WORK_DIR, 'inference_results_iq.pkl')

def get_person_count(vec):
    """
    根据向量判断人数类别: 1, 2
    vec: [d1, a1, d2, a2]
    """
    d2 = vec[:, 2] 
    count = np.ones(len(vec), dtype=int)
    # 只要第二个距离不是 -1，就认为是双人
    count[d2 != -1] = 2
    return count

print(f"Loading results from {RESULT_FILE}...")
with open(RESULT_FILE, 'rb') as f: 
    data = pickle.load(f)
    
y_true = data['y_test'].astype(int)
y_pred = data['y_pred'].astype(int)

# ================= 1. 总体全匹配准确率 =================
exact_matches = np.all(y_true == y_pred, axis=1)
total_acc = np.mean(exact_matches)

print(f"\n==========================================")
print(f"TOTAL EXACT ACCURACY (End-to-End Classification)")
print(f"==========================================")
print(f"Total Samples: {len(y_true)}")
print(f"Exact Matches: {np.sum(exact_matches)}")
print(f"Accuracy:      {total_acc*100:.2f}%")
print(f"\nSample Prediction:\n{y_pred[:5]}")
print(f"Sample Truth:\n{y_true[:5]}")

# ================= 2. 人数检测准确率 =================
y_true_cnt = get_person_count(y_true)
y_pred_cnt = get_person_count(y_pred)

print(f"\n------------------------------------------")
print(f"PERSON COUNT METRICS (Single vs Double)")
print(f"------------------------------------------")
print(classification_report(y_true_cnt, y_pred_cnt, labels=[1, 2], target_names=['Single', 'Double']))

cm = confusion_matrix(y_true_cnt, y_pred_cnt, labels=[1, 2])
print("Confusion Matrix (Row=True, Col=Pred):")
print(f"                 Pred Single  Pred Double")
print(f"True Single      {cm[0][0]:<12} {cm[0][1]}")
print(f"True Double      {cm[1][0]:<12} {cm[1][1]}")

# ================= 3. 错误案例分析 =================
if not np.all(exact_matches):
    print(f"\n------------------------------------------")
    print(f"TOP 5 CONFUSION TYPES (True -> Predicted)")
    print(f"------------------------------------------")
    err_indices = np.where(~exact_matches)[0]
    
    error_counts = {}
    for idx in err_indices:
        t_str = str(tuple(y_true[idx]))
        p_str = str(tuple(y_pred[idx]))
        key = f"{t_str} -> {p_str}"
        error_counts[key] = error_counts.get(key, 0) + 1
    
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_errors[:5]:
        print(f"{k}: {v} times")
else:
    print("\nPerfect Prediction! No errors found.")