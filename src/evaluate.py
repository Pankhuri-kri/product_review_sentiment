"""
Shared evaluation + comparison-table builder.
Owner: Sai Prasad Prusty (230911298)
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    result = {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    print(f"{name:14s} | acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    return result


def build_comparison_table(results, out_path="preliminary_results.csv"):
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print("\n=== Comparison table (preliminary) ===")
    print(res_df.to_string(index=False))
    return res_df
