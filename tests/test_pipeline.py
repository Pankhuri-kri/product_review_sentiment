"""Regression checks for sampling, leakage, bounded densification, and metrics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_utils import select_balanced_reviews, audit_splits, prepare
from evaluate import evaluate, cue_flags, export_evaluation
from pipeline import parse_args, run_stage
from preprocessing import clean_text
from split_utils import build_splits
from training_utils import dense_batches


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = [{"label": i % 2, "title": f"Review {i}", "content": f"This product has source number {i}."} for i in range(240)]
        self.frame = pd.DataFrame(rows)
        for k, part in enumerate([self.frame.iloc[:85], self.frame.iloc[85:]]):
            pq.write_table(pa.Table.from_pandas(part, preserve_index=False), self.root / f"train-{k:05d}-of-00002.parquet")
        self.files = sorted(self.root.glob("*.parquet"))

    def tearDown(self):
        self.temp.cleanup()

    def test_sampler_matches_original_uniform_sampling_without_full_pandas(self):
        actual, _ = select_balanced_reviews(self.files, 100, 42, batch_size=7)
        expected = pd.concat([self.frame[self.frame.label == label].sample(n=50, random_state=42) for label in (1, 0)]).sample(frac=1, random_state=42)
        self.assertEqual(actual.review_id.tolist(), expected.index.tolist())
        self.assertEqual(actual.label.value_counts().to_dict(), {0:50, 1:50})
        again, _ = select_balanced_reviews(self.files, 100, 42, batch_size=19)
        pd.testing.assert_frame_equal(actual, again)

    def test_duplicates_are_replaced_before_splitting(self):
        frame = self.frame.copy()
        frame.loc[2, ["title","content"]] = frame.loc[0, ["title","content"]].values
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), self.root / "duplicates.parquet")
        selected, audit = select_balanced_reviews([self.root / "duplicates.parquet"], 100, 42, 7)
        self.assertEqual(selected.text.nunique(), 100)
        self.assertGreater(audit["candidate_duplicates_removed"], 0)
        self.assertEqual(sorted(selected.label.value_counts()), [50,50])

    def test_invalid_label_and_sample_count_fail(self):
        with self.assertRaises(ValueError):
            select_balanced_reviews(self.files, 99)
        bad = self.frame.copy()
        bad.loc[0,"label"] = 2
        path = self.root / "invalid.parquet"
        pq.write_table(pa.Table.from_pandas(bad, preserve_index=False), path)
        with self.assertRaises(ValueError):
            select_balanced_reviews([path], 100)

    def test_stratified_disjoint_splits_and_leakage_detection(self):
        df, _ = select_balanced_reviews(self.files, 100)
        train, validation, test = build_splits(df)
        audit = audit_splits(dict(train=train, validation=validation, test=test))
        self.assertEqual([audit[k]["rows"] for k in audit], [80,10,10])
        with self.assertRaises(ValueError):
            audit_splits(dict(train=train, test=train.copy()))

    def test_sparse_batching_densifies_only_a_batch(self):
        features = sparse.eye(103, 20000, format="csr", dtype=np.float32)
        labels = np.arange(103)
        shapes = []
        original = sparse.csr_matrix.toarray
        def guarded(matrix, *args, **kwargs):
            shapes.append(matrix.shape)
            self.assertLessEqual(matrix.shape[0], 16)
            return original(matrix, *args, **kwargs)
        with patch.object(sparse.csr_matrix, "toarray", guarded):
            batches = list(dense_batches(features, labels, 16))
        self.assertEqual(sum(len(x) for x,y in batches), 103)
        np.testing.assert_array_equal(np.concatenate([y for x,y in batches]), labels)
        self.assertEqual(max(x.nbytes for x,y in batches), 16*20000*4)
        self.assertEqual(len(shapes), 7)

    def test_sparse_float32_artifacts_and_train_only_vectorizer(self):
        audit = prepare(self.root / "prepared", 100, shard_dir=self.root)
        features = sparse.load_npz(self.root / "prepared/train_tfidf.npz")
        self.assertEqual(features.dtype, np.float32)
        self.assertLess(audit["tfidf"]["train"]["csr_bytes"], audit["tfidf"]["train"]["hypothetical_dense_float64_bytes"])
        import pickle
        with (self.root / "prepared/vectorizer.pkl").open("rb") as stream:
            vectorizer = pickle.load(stream)
        train = pd.read_csv(self.root / "prepared/train.csv")
        train_terms = set(term for text in train.text for term in vectorizer.build_analyzer()(text))
        self.assertTrue(set(vectorizer.vocabulary_).issubset(train_terms))

    def test_common_metrics_and_error_exports(self):
        result = evaluate("test", [0,0,1,1], [0,1,1,1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["f1"], 0.8)
        test = pd.DataFrame(dict(review_id=[1,2,3,4], raw_text=["bad", "not good", "great but broken", "yeah right"], text=["bad", "not good", "great but broken", "yeah right"], label=[0,0,1,1]))
        export_evaluation("mlp", test, [.1,.8,.9,.7], self.root / "evaluation")
        errors = pd.read_csv(self.root / "evaluation/misclassified.csv")
        self.assertEqual(errors.review_id.tolist(), [2])
        self.assertTrue(errors.negation_cue.iloc[0])
        self.assertTrue(cue_flags("Great design but terrible battery")["mixed_sentiment_cue"])
        self.assertTrue(cue_flags("Yeah right, just great")["sarcasm_cue"])
        with self.assertRaises(ValueError):
            export_evaluation("mlp", test, [.1,.2], self.root / "bad")

    def test_real_defaults_and_negation_cleaning(self):
        config = parse_args([])
        self.assertEqual(config.samples, 20000)
        self.assertFalse(hasattr(config, "synthetic"))
        self.assertEqual(clean_text("<b>NOT good!</b> https://example.com I wouldn't buy."), "not good! i wouldn't buy.")

    def test_resume_checks_output_hashes(self):
        (self.root / "logs").mkdir()
        output = self.root / "output.txt"
        script = self.root / "write.py"
        script.write_text("from pathlib import Path\nPath("+repr(str(output))+").write_text('valid')", encoding="utf-8")
        first = run_stage("test", [script], [output], self.root, "signature", False)
        self.assertEqual(run_stage("test", [script], [output], self.root, "signature", True), first)
        output.write_text("corrupt", encoding="utf-8")
        run_stage("test", [script], [output], self.root, "signature", True)
        self.assertEqual(output.read_text(), "valid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
