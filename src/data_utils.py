"""Read Amazon parquet in bounded batches and prepare shared sparse features."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import clean_text
from split_utils import build_splits

REVISION = "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"


def resolve_shards(revision=REVISION, shard_dir=None):
    if shard_dir:
        files = sorted(Path(shard_dir).glob("train-*.parquet"))
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        kwargs = dict(repo_id="fancyzhx/amazon_polarity", repo_type="dataset", revision=revision,
                      allow_patterns=["amazon_polarity/train-*.parquet"])
        try:
            folder = snapshot_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            folder = snapshot_download(**kwargs)
        files = sorted((Path(folder) / "amazon_polarity").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError("No Amazon Polarity training parquet shards found")
    return files


def select_balanced_reviews(files, n_samples=20000, seed=42, batch_size=1024):
    """Sample source row IDs first; only selected strings reach pandas.

    The full label vector uses 3.6 MB as int8. Review strings are read in
    batches, with deterministic reserves replacing duplicate or empty reviews.
    There is no synthetic-data fallback.
    """
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples must be positive and even")
    population = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    labels = np.empty(population, dtype=np.int8)
    offset = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["label"]):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            labels[offset:offset+len(values)] = values
            offset += len(values)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Dataset contains non-binary labels")
    wanted = n_samples // 2
    candidate_ids = []
    for label in (1, 0):
        ids = np.flatnonzero(labels == label)
        if len(ids) < wanted:
            raise ValueError(f"Not enough label {label} rows for {n_samples} samples")
        reserve = min(len(ids), wanted + max(100, wanted // 10))
        candidate_ids.extend(ids[np.random.RandomState(seed).choice(len(ids), reserve, replace=False)].tolist())
    selected = np.sort(np.asarray(candidate_ids, dtype=np.int64))
    rows = {}
    offset = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["title", "content"]):
            end = offset + batch.num_rows
            left, right = np.searchsorted(selected, [offset, end])
            if right > left:
                ids = selected[left:right]
                for row_id, record in zip(ids, batch.take((ids-offset).astype(np.int32)).to_pylist()):
                    raw = (record["title"] or "") + ". " + (record["content"] or "")
                    rows[int(row_id)] = {"review_id": int(row_id), "raw_text": raw,
                                         "text": clean_text(raw), "label": int(labels[row_id])}
            offset = end
    if len(rows) != len(selected):
        raise ValueError("Some sampled source rows were not read")
    candidates = pd.DataFrame([rows[i] for i in candidate_ids])
    conflicts = candidates.groupby("text").label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    valid = candidates[~candidates.text.isin(conflicts) & candidates.text.ne("")].drop_duplicates("text")
    balanced = [valid[valid.label == label].head(wanted) for label in (1, 0)]
    if any(len(part) != wanted for part in balanced):
        raise ValueError("Insufficient unique reviews after cleaning; increase the sampling reserve")
    subset = pd.concat(balanced).sample(frac=1, random_state=seed).reset_index(drop=True)
    audit = {"population_rows": population, "requested_samples": n_samples,
             "sampled_candidates": len(candidates), "candidate_duplicates_removed": int(candidates.text.duplicated().sum()),
             "conflicting_cleaned_texts_excluded": len(conflicts), "unique_reviews": int(subset.text.nunique()),
             "label_counts": {str(k): int(v) for k,v in subset.label.value_counts().items()},
             "sampling": "Uniform without replacement per label; deterministic reserves replace empty, duplicate, or conflicting cleaned reviews",
             "source_row_ids_sha256": hashlib.sha256(subset.review_id.to_numpy(dtype=np.int64).tobytes()).hexdigest()}
    return subset, audit


def audit_splits(parts):
    audit = {}
    for name, df in parts.items():
        if df.review_id.duplicated().any() or df.text.duplicated().any():
            raise ValueError(f"Duplicate review in {name}")
        counts = df.label.value_counts().to_dict()
        if counts.get(0, 0) != counts.get(1, 0):
            raise ValueError(f"Unbalanced labels in {name}")
        audit[name] = {"rows": len(df), "label_counts": {str(k):int(v) for k,v in counts.items()}}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if set(parts[a].review_id) & set(parts[b].review_id) or set(parts[a].text) & set(parts[b].text):
                raise ValueError(f"Data leakage between {a} and {b}")
    return audit


def prepare(output_dir, n_samples=20000, seed=42, revision=REVISION, shard_dir=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = resolve_shards(revision, shard_dir)
    df, audit = select_balanced_reviews(files, n_samples, seed)
    train, validation, test = build_splits(df, seed)
    parts = dict(train=train, validation=validation, test=test)
    audit["splits"] = audit_splits(parts)
    audit.update(dataset="fancyzhx/amazon_polarity", revision=revision, seed=seed,
                 source_files=[{"name":p.name,"bytes":p.stat().st_size} for p in files],
                 source="https://huggingface.co/datasets/fancyzhx/amazon_polarity/tree/"+revision,
                 text_batch_size=1024, source_partition="train only; official test partition not used")
    df.to_csv(root / "subset.csv", index=False)
    for name, part in parts.items():
        part.to_csv(root / f"{name}.csv", index=False)
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), dtype=np.float32)
    audit["tfidf"] = {}
    for name, part in parts.items():
        features = vectorizer.fit_transform(part.text) if name == "train" else vectorizer.transform(part.text)
        features = features.tocsr()
        sparse.save_npz(root / f"{name}_tfidf.npz", features)
        audit["tfidf"][name] = {"shape": list(features.shape), "dtype": str(features.dtype), "nnz": int(features.nnz),
                                 "csr_bytes": int(features.data.nbytes+features.indices.nbytes+features.indptr.nbytes),
                                 "hypothetical_dense_float64_bytes": int(np.prod(features.shape))*8}
    with (root / "vectorizer.pkl").open("wb") as stream:
        pickle.dump(vectorizer, stream)
    (root / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--shard-dir", type=Path)
    args = parser.parse_args()
    prepare(args.output_dir, args.samples, args.seed, args.revision, args.shard_dir)
