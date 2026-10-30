# Product review sentiment classification

ICT4442 project comparing **MLP, TextCNN, and BiLSTM** on **20,000 real Amazon Polarity reviews**. Real data is the default; there is no synthetic fallback.

## Models

| Model | Input and architecture | Owner |
| --- | --- | --- |
| MLP | TF-IDF unigrams/bigrams; Dense 256 and 64 | Pankhuri Kumari |
| TextCNN | Learned embeddings; 128 Conv1D filters, width 5; max pooling | Sai Prasad Prusty |
| BiLSTM | Learned embeddings; 64 LSTM units per direction; masked padding | Jambhorkar Arya Sachin |

## Dataset and training

- **Data:** 10,000 positive and 10,000 negative reviews from `fancyzhx/amazon_polarity`, pinned revision `9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`.
- **Common protocol:** seed 42; identical cleaning and stratified 16,000/2,000/2,000 train/validation/test split; no shared source IDs or duplicate cleaned reviews across splits.
- **Features:** training-only TF-IDF/tokenizer fitting; vocabulary cap 20,000. TextCNN/BiLSTM share 100-token sequences and 100-dimensional embeddings.
- **Tuning:** Adam, weight decay 0.0001, batch size 32, maximum 8 epochs, early stopping with patience 2. The same two dropout/learning-rate pairs are tried for every architecture: dropout/learning rate **0.3/0.001** and **0.5/0.0005**.
- **Selection:** best checkpoint and candidate chosen by validation loss only, followed by evaluation on the common test set at threshold 0.5.

## Setup and run

Use **Python 3.12**. Run these PowerShell commands from this project folder:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' src/pipeline.py --resume
```

In the existing DL Project workspace, use `..\.verification-env\Scripts\python.exe` instead of creating another environment. The first run downloads public training shards unless cached; `--shard-dir PATH` accepts existing Amazon training parquet files.

**Safe reruns:** `--resume` verifies signatures and output hashes before reusing completed stages. Changes to source/settings can trigger retraining and replace outputs. Preserve this run by choosing a new folder for experiments:

```powershell
& '.\.venv\Scripts\python.exe' src/pipeline.py --output-dir results/new_run
```

## Jupyter Notebook

The entire project is in [DL_Project_Complete.ipynb](DL_Project_Complete.ipynb), at this repository's root. It contains the real 20,000-review dataset, all model/training/evaluation code, all six trained candidates' inference weights, 14 visualization figures, and 41 original source/test/report attachments. Run all cells with `MODE = "saved"` to restore the included models, recompute test results, and predict new reviews; `MODE = "train"` retrains in memory. It uses no companion project files and creates no separate CSV/model/image files. Required Python packages and setup instructions are inside the notebook.

The original script-based notebook is also retained:

Install and launch Jupyter in the same environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install notebook ipykernel
& '.\.venv\Scripts\python.exe' -m ipykernel install --sys-prefix --name amazon-verification
& '.\.venv\Scripts\python.exe' -m notebook notebook/DL_Phase2_Sentiment_Pipeline_EXECUTED.ipynb
```

Open the [notebook](notebook/DL_Phase2_Sentiment_Pipeline_EXECUTED.ipynb), select the `amazon-verification` kernel, and choose **Run → Run All Cells**. It calls the shared pipeline with `--resume` and displays results, memory usage, and failure cases.

## Real test results

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MLP | **88.20%** | 0.8760 | 0.8900 | **0.8829** |
| TextCNN | 85.40% | 0.8371 | 0.8790 | 0.8576 |
| BiLSTM | 86.10% | 0.8588 | 0.8640 | 0.8614 |

Precision, recall, and F1 refer to the positive class. These are single-seed results; early stopping controls overfitting but does not eliminate it.

## Memory and outputs

Batched parquet reading, float32 sparse TF-IDF, batch-only densification, memory-mapped sequences, and separate worker processes reduce memory use. Training TF-IDF storage is **10.6 MB versus 2.56 GB** previously (about **241× smaller**); peak measured private memory was **686.5 MiB** per stage on CPU.

Everything is saved under `results/amazon_20k/`:

| Output | Contents |
| --- | --- |
| [RESULTS.md](results/amazon_20k/RESULTS.md), [comparison.csv](results/amazon_20k/comparison.csv) | Scores, methodology, architectural discussion, and failure examples |
| [ERROR_REVIEW.md](results/amazon_20k/ERROR_REVIEW.md) | Nine reviewed cases: negation, mixed sentiment, irony, language limitations, and a suspected label/text conflict |
| `models/` | Best checkpoints, settings, and training histories for all six candidates |
| `evaluation/` | Predictions, misclassified reviews, class reports, confusion matrices, and heuristic cue summaries |
| `data_audit.json`, `run_configuration.json`, `resource_usage.json` | Dataset/split provenance, source hashes, configuration, memory, and timing |

Cue flags are review aids, not verified sarcasm annotations. Qualitative reviews accompany the completed experiment; new runs generate automatic failure exports. Long reviews are truncated to 100 tokens for sequence models.

## Checks and Git

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Verified: **9 regression tests**, **38 independent real-result checks**, and **all 6 notebook code cells** passed. `.gitignore` excludes environments, datasets, checkpoints, feature arrays, logs, resume markers, and full prediction dumps; source, notebooks, reports, metrics, histories, and reviewed cases remain available to commit.
