"""Common metrics, auditable failure cases, and the final comparison report.

Owner: Sai Prasad Prusty (230911298).
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

MODEL_NAMES = {"mlp":"MLP (TF-IDF)", "textcnn":"TextCNN", "bilstm":"BiLSTM"}


def evaluate(name, y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result = {"model": name, "accuracy": float(accuracy_score(y_true, y_pred)),
              "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    print(f"{name:14s} | acc={result['accuracy']:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}", flush=True)
    return result


def build_comparison_table(results, out_path="comparison.csv"):
    table = pd.DataFrame(results)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False), flush=True)
    return table


def cue_flags(text):
    """Nonexclusive review aids, never treated as ground-truth error types."""
    text = str(text).lower()
    words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", text))
    positive = {"good","great","love","excellent","perfect","amazing","best","happy","recommend","nice"}
    negative = {"bad","terrible","awful","worst","broken","broke","poor","disappointed","waste","useless","horrible"}
    return {
        "negation_cue": bool(words & {"not","no","never","nothing","neither","nor","hardly","without","don't","doesn't","didn't","isn't","wasn't","wouldn't","can't","couldn't","won't"}),
        "mixed_sentiment_cue": bool(words & positive and words & negative),
        "sarcasm_cue": bool(re.search(r"yeah right|thanks for nothing|what a joke|just great|as if|so much for", text)),
    }


def export_evaluation(name, test, probabilities, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(probabilities) != len(test) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Expected one finite probability in [0,1] per test review")
    predictions = (probabilities > 0.5).astype(np.int8)
    metrics = evaluate(MODEL_NAMES.get(name, name), test.label.to_numpy(), predictions)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output = test.copy()
    output["positive_probability"] = probabilities
    output["prediction"] = predictions
    output["correct"] = output.label == output.prediction
    flags = pd.DataFrame([cue_flags(t) for t in output.text], index=output.index)
    output = pd.concat([output, flags], axis=1)
    output["confidence"] = np.maximum(probabilities, 1-probabilities)
    output.to_csv(folder / "predictions.csv", index=False)
    errors = output[~output.correct].sort_values("confidence", ascending=False)
    errors.to_csv(folder / "misclassified.csv", index=False)
    report = classification_report(output.label, output.prediction, labels=[0,1], target_names=["negative","positive"], output_dict=True, zero_division=0)
    (folder / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(output.label, output.prediction, labels=[0,1]),
                 index=["true_negative","true_positive"], columns=["predicted_negative","predicted_positive"]).to_csv(folder / "confusion_matrix.csv")
    categories, examples = [], []
    for category in flags.columns:
        subset = output[output[category]]
        failures = errors[errors[category]]
        categories.append({"category": category, "reviews": len(subset), "errors": len(failures),
                           "error_rate": float((~subset.correct).mean()) if len(subset) else None,
                           "annotation": "heuristic cue, not a manually verified sentiment phenomenon"})
        excerpt = failures.head(5).copy()
        excerpt.insert(0, "review_category", category)
        examples.append(excerpt)
    other = errors[~errors[flags.columns].any(axis=1)].head(5).copy()
    other.insert(0, "review_category", "other")
    examples.append(other)
    pd.DataFrame(categories).to_csv(folder / "error_categories.csv", index=False)
    pd.concat(examples, ignore_index=True).to_csv(folder / "failure_examples.csv", index=False)
    return metrics


def markdown_table(frame, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], (float,np.floating)) else str(row[c]) for c in columns) + " |")
    return "\n".join(rows)


def write_report(root):
    root = Path(root)
    selected = json.loads((root / "selected_models.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
    config = json.loads((root / "run_configuration.json").read_text(encoding="utf-8"))
    results = [json.loads((root / "evaluation" / key / "metrics.json").read_text(encoding="utf-8")) for key in MODEL_NAMES]
    table = build_comparison_table(results, root / "comparison.csv")
    best = table.loc[table.f1.idxmax()]
    lines = ["# Real Amazon Polarity results", "", f"All three architectures completed the same {config['samples']:,}-review experiment. {best['model']} had the highest positive-class F1 in this run ({best.f1:.4f}).", "",
             markdown_table(table, ["model","accuracy","precision","recall","f1"]), "", "## Common experimental protocol", "",
             f"Source: [Amazon Polarity]({audit['source']}), revision `{config['revision']}`. Label 0 is negative; label 1 is positive. The subset has {audit['unique_reviews']:,} distinct cleaned reviews, balanced across labels. Empty, duplicate, and conflicting cleaned texts are excluded before splitting. All models share source review IDs, cleaning, labels, and the seed-{config['seed']} stratified 80/10/10 split (16,000/2,000/2,000 for the default 20,000-row run).", "",
             "TF-IDF unigrams and bigrams (up to 20,000 features) are fitted on training reviews only for the MLP. The shared training-only tokenizer produces 100-token sequences and learned 100-dimensional embeddings for TextCNN and BiLSTM. Long reviews are truncated, which can remove sentiment cues near the end. BiLSTM masks padded tokens.", "",
             f"Both candidates for every model used batch size {config['batch_size']}, Adam with weight decay, at most {config['epochs']} epochs, and early stopping on validation loss with patience {config['patience']}. The identical search grid changes dropout and learning rate. The best checkpoint and candidate are selected only by validation loss. Each selected model is then evaluated once on the held-out test set; the decision threshold is fixed at 0.5. Precision, recall, and F1 refer to the positive class.", "", "## Validation selection and overfitting", ""]
    tuning = []
    for key, name in MODEL_NAMES.items():
        s = selected[key]
        tuning.append({"model":name,"candidate":s["candidate"],"epochs":s["trained_epochs"],"best_epoch":s["best_epoch"],"validation_loss":s["best_val_loss"],"train_accuracy":s["best_train_accuracy"],"validation_accuracy":s["best_val_accuracy"]})
    lines += [markdown_table(pd.DataFrame(tuning),list(tuning[0])), "",
              "Early stopping and best-checkpoint restoration prevent reporting a later, worse validation epoch. They do not eliminate overfitting: the training/validation accuracy gap at the selected epoch remains a useful diagnostic. Histories for both candidates are retained under `models/`; test scores were not used to change the grid or choose epochs.", "", "## Architectural interpretation", "",
              "The MLP sees sparse unigram and bigram sentiment cues throughout the review but has limited explicit order information. TextCNN pools local patterns from the first 100 tokens. BiLSTM models sequential context in both directions over those same tokens. Learned embeddings start from scratch, and the dataset size, sequence truncation, optimizer settings, and training dynamics can all affect the ranking. This single-seed run is evidence about these configurations; it does not establish that one architecture is universally superior.", "", "## Failure analysis", "",
              "Every test prediction and misclassified review is exported with source row ID, raw review, true label, predicted label, and positive probability. Negation, mixed-word-polarity, and possible-sarcasm flags are nonexclusive heuristic review aids, not ground-truth annotations. In particular, the sarcasm error rate cannot be inferred from these cue counts.", ""]
    for key, name in MODEL_NAMES.items():
        folder = root / "evaluation" / key
        errors = pd.read_csv(folder / "misclassified.csv")
        categories = pd.read_csv(folder / "error_categories.csv")
        lines += [f"### {name}", "", f"Misclassified {len(errors)} of {audit['splits']['test']['rows']} test reviews.", "",
                  markdown_table(categories, ["category","reviews","errors","error_rate"]), ""]
        examples = pd.read_csv(folder / "failure_examples.csv").drop_duplicates("review_id").head(3)
        for _, row in examples.iterrows():
            excerpt = re.sub(r"\s+", " ", row.raw_text)[:500].replace("|", "\\|")
            lines += [f"- Source row {row.review_id}: true={row.label}, predicted={row.prediction}, P(positive)={row.positive_probability:.4f}. Review excerpt: {excerpt}"]
        lines += ["", f"Full records: `evaluation/{key}/predictions.csv`, `misclassified.csv`, `failure_examples.csv`, `classification_report.json`, and `confusion_matrix.csv`.", ""]
    lines += ["## Memory use and reproducibility", "",
              "The loader scans parquet in batches instead of converting the multi-million-row corpus to pandas. Float32 TF-IDF matrices remain CSR sparse; only one 32-row batch is densified at a time (about 2.44 MiB at 20,000 features). TensorFlow stages run in separate processes, releasing allocations between candidates. Sequence arrays are loaded with memory mapping. `data_audit.json` records sparse storage sizes, and `resource_usage.json` records measured stage memory and duration.", "",
              "The dataset revision, sampled source IDs, split CSVs, source hashes, package versions, tuning histories, checkpoints, and predictions are saved. `--resume` verifies stage signatures and output hashes before reuse. Results can vary across library versions or hardware; repeated seeds and confidence intervals remain future work.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    write_report(parser.parse_args().output_dir)
