"""One TensorFlow operation per process; test data is never used for tuning."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf

from preprocessing import build_tokenizer, texts_to_padded_sequences, vocab_size_of
from training_utils import fit_model, predict_batches


def prepare_sequences(root):
    train = pd.read_csv(root / "train.csv", usecols=["text"])
    tokenizer = build_tokenizer(train.text)
    (root / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {"vocab_size": vocab_size_of(tokenizer), "max_length": 100, "embedding_dim": 100, "splits": {}}
    for name in ("train", "validation", "test"):
        frame = train if name == "train" else pd.read_csv(root / f"{name}.csv", usecols=["text"])
        sequences = texts_to_padded_sequences(tokenizer, frame.text)
        np.save(root / f"{name}_sequences.npy", sequences)
        lengths = np.fromiter((len(s) for s in tokenizer.texts_to_sequences(frame.text)), dtype=np.int32)
        metadata["splits"][name] = {"shape": list(sequences.shape), "dtype": str(sequences.dtype),
                                    "bytes": int(sequences.nbytes), "truncated_reviews": int((lengths>100).sum())}
    (root / "sequence_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def load_features(root, model, split):
    if model == "mlp":
        return sparse.load_npz(root / f"{split}_tfidf.npz")
    return np.load(root / f"{split}_sequences.npy", mmap_mode="r")


def train(root, name, candidate_name):
    config = json.loads((root / "run_configuration.json").read_text(encoding="utf-8"))
    settings = next(c for c in config["candidates"] if c["name"] == candidate_name)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    features = load_features(root, name, "train")
    validation = load_features(root, name, "validation")
    labels = pd.read_csv(root / "train.csv", usecols=["label"]).label.to_numpy()
    val_labels = pd.read_csv(root / "validation.csv", usecols=["label"]).label.to_numpy()
    kwargs = {k:v for k,v in settings.items() if k != "name"}
    if name == "mlp":
        from mlp_model import build_mlp
        model = build_mlp(features.shape[1], **kwargs)
    else:
        vocab = json.loads((root / "sequence_metadata.json").read_text(encoding="utf-8"))["vocab_size"]
        if name == "textcnn":
            from textcnn_model import build_textcnn
            model = build_textcnn(vocab, **kwargs)
        else:
            from bilstm_model import build_bilstm
            model = build_bilstm(vocab, **kwargs)
    folder = root / "models" / name / candidate_name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model.summary()
    history = fit_model(model, features, labels, validation, val_labels, epochs=config["epochs"],
                        batch_size=config["batch_size"], patience=config["patience"], seed=config["seed"], checkpoint=folder / "model.keras")
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame)+1))
    frame.to_csv(folder / "history.csv", index=False)
    best = int(frame.val_loss.to_numpy().argmin())
    summary = {"model": name, "candidate": candidate_name, "settings": settings,
               "trained_epochs": len(frame), "best_epoch": best+1, "best_val_loss": float(frame.val_loss.iloc[best]),
               "best_val_accuracy": float(frame.val_accuracy.iloc[best]), "best_train_accuracy": float(frame.accuracy.iloc[best]),
               "seconds": round(time.monotonic()-started, 2), "parameter_count": model.count_params(),
               "test_used_for_selection": False, "signature": config["signature"],
               "versions": {p:importlib.metadata.version(p) for p in ("tensorflow","keras","numpy","scikit-learn")},
               "devices": [str(d) for d in tf.config.list_physical_devices()]}
    (folder / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(root, name, candidate):
    from evaluate import export_evaluation
    config = json.loads((root / "run_configuration.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(root / "models" / name / candidate / "model.keras", compile=False)
    test = pd.read_csv(root / "test.csv")
    features = load_features(root, name, "test")
    probabilities = predict_batches(model, features, config["batch_size"])
    export_evaluation(name, test, probabilities, root / "evaluation" / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare-sequences", "train", "evaluate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("mlp","textcnn","bilstm"))
    parser.add_argument("--candidate", choices=("standard","regularized"))
    args = parser.parse_args()
    if args.operation == "prepare-sequences":
        prepare_sequences(args.output_dir)
    elif args.operation == "train":
        train(args.output_dir, args.model, args.candidate)
    else:
        evaluate(args.output_dir, args.model, args.candidate)
