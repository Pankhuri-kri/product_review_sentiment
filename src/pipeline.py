"""Real Amazon Polarity experiment with isolated, memory-bounded stages.

Run from the project directory: python src/pipeline.py --resume
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REVISION = "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"
MODELS = ("mlp", "textcnn", "bilstm")
CANDIDATES = (
    {"name": "standard", "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0001},
    {"name": "regularized", "dropout": 0.5, "learning_rate": 0.0005, "weight_decay": 0.0001},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/amazon_20k")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--shard-dir", type=Path, help="Optional local Amazon training parquet directory")
    parser.add_argument("--resume", action="store_true", help="Reuse only stages with matching signatures and output hashes")
    args = parser.parse_args(argv)
    if args.samples < 100 or args.samples % 2:
        parser.error("--samples must be an even integer >= 100")
    if min(args.epochs, args.batch_size) < 1 or args.patience < 0:
        parser.error("epochs and batch size must be positive; patience must be nonnegative")
    return args


def run_stage(name, argv, outputs, root, signature, resume):
    import psutil
    marker = root / f".{name}.complete.json"
    if resume and marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("signature") == signature and old.get("files") == {
            str(p.relative_to(root)): sha256(p) for p in outputs if p.exists()
        } and all(p.exists() for p in outputs):
            print(f"REUSE {name}: verified completed outputs", flush=True)
            return old
    marker.unlink(missing_ok=True)
    log = root / "logs" / f"{name}.log"
    print(f"START {name}; log: {log}", flush=True)
    started = time.monotonic()
    peak_private = peak_rss = low_memory_samples = 0
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, "-u", *map(str, argv)], stdout=stream, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            private = rss = 0
            try:
                for member in [process, *process.children(recursive=True)]:
                    try:
                        info = member.memory_info()
                        private += getattr(info, "private", info.rss)
                        rss += info.rss
                    except psutil.NoSuchProcess:
                        pass
                peak_private, peak_rss = max(peak_private, private), max(peak_rss, rss)
                low_memory_samples = low_memory_samples + 1 if psutil.virtual_memory().available < 128 * 2**20 else 0
                if private > 4 * 2**30 or low_memory_samples >= 3:
                    for member in process.children(recursive=True):
                        try:
                            member.terminate()
                        except psutil.Error:
                            pass
                    child.terminate()
                    child.wait(timeout=30)
                    raise RuntimeError(f"{name}: memory guard stopped this stage; close unused applications and rerun --resume. See {log}")
            except psutil.NoSuchProcess:
                pass
            time.sleep(1)
        code = child.wait()
    if code:
        print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
        raise RuntimeError(f"{name} failed with exit code {code}; see {log}")
    if any(not p.exists() for p in outputs):
        raise RuntimeError(f"{name} did not produce all expected files")
    result = {"signature": signature, "seconds": round(time.monotonic()-started, 2),
              "peak_private_mib": round(peak_private / 2**20, 1), "peak_rss_mib": round(peak_rss / 2**20, 1),
              "files": {str(p.relative_to(root)): sha256(p) for p in outputs}}
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS {name}: {result['seconds']}s; peak private {result['peak_private_mib']} MiB", flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    root = args.output_dir.resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    cache = PROJECT.parent / ".verification-cache/huggingface"
    if cache.exists():
        os.environ.setdefault("HF_HOME", str(cache))
    for key, value in {"PYTHONIOENCODING":"utf-8", "TF_CPP_MIN_LOG_LEVEL":"2",
                       "TF_NUM_INTRAOP_THREADS":"2", "TF_NUM_INTEROP_THREADS":"1",
                       "OMP_NUM_THREADS":"2", "HF_HUB_DISABLE_PROGRESS_BARS":"1"}.items():
        os.environ.setdefault(key, value)
    config = {"dataset": "fancyzhx/amazon_polarity", "revision": args.revision,
              "samples": args.samples, "seed": args.seed, "epochs": args.epochs,
              "patience": args.patience, "batch_size": args.batch_size,
              "max_vocab": 20000, "max_length": 100, "embedding_dim": 100,
              "candidates": CANDIDATES, "shard_dir": str(args.shard_dir.resolve()) if args.shard_dir else None,
              "source_hashes": {p.name: sha256(p) for p in sorted((PROJECT / "src").glob("*.py"))}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["signature"] = signature
    (root / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    stages = {}
    prep = [PROJECT / "src/data_utils.py", "--output-dir", root, "--samples", args.samples,
            "--seed", args.seed, "--revision", args.revision]
    if args.shard_dir:
        prep += ["--shard-dir", args.shard_dir.resolve()]
    prepared = [root / n for n in ("train.csv", "validation.csv", "test.csv", "subset.csv", "data_audit.json", "vectorizer.pkl",
                                  "train_tfidf.npz", "validation_tfidf.npz", "test_tfidf.npz")]
    stages["prepare-data"] = run_stage("prepare-data", prep, prepared, root, signature, args.resume)
    worker = PROJECT / "src/train_worker.py"
    sequence_outputs = [root / n for n in ("tokenizer.json", "sequence_metadata.json", "train_sequences.npy", "validation_sequences.npy", "test_sequences.npy")]
    stages["prepare-sequences"] = run_stage("prepare-sequences", [worker, "prepare-sequences", "--output-dir", root], sequence_outputs, root, signature, args.resume)
    selected = {}
    for model in MODELS:
        candidates = []
        for candidate in CANDIDATES:
            name = f"train-{model}-{candidate['name']}"
            folder = root / "models" / model / candidate["name"]
            outputs = [folder / p for p in ("model.keras", "history.csv", "training_summary.json")]
            stages[name] = run_stage(name, [worker, "train", "--output-dir", root, "--model", model,
                                           "--candidate", candidate["name"]], outputs, root, signature, args.resume)
            candidates.append(json.loads(outputs[-1].read_text(encoding="utf-8")))
        selected[model] = min(candidates, key=lambda c: c["best_val_loss"])
    (root / "selected_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    # Candidates never open test features or labels. Select using validation only.
    for model in MODELS:
        folder = root / "evaluation" / model
        outputs = [folder / p for p in ("metrics.json", "predictions.csv", "misclassified.csv", "error_categories.csv", "failure_examples.csv", "classification_report.json", "confusion_matrix.csv")]
        name = f"evaluate-{model}"
        stages[name] = run_stage(name, [worker, "evaluate", "--output-dir", root, "--model", model,
                                       "--candidate", selected[model]["candidate"]], outputs, root, signature, args.resume)
    stages["report"] = run_stage("report", [PROJECT / "src/evaluate.py", "--output-dir", root],
                             [root / "comparison.csv", root / "RESULTS.md"], root, signature, args.resume)
    (root / "resource_usage.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
    print("\nREAL AMAZON EXPERIMENT COMPLETED\n" + (root / "comparison.csv").read_text(encoding="utf-8"), flush=True)
    return root


if __name__ == "__main__":
    main()
