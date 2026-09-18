"""
ICT-4442 Deep Learning Mini Project — Phase 2 (Interim)
Product Review Sentiment Classification: MLP vs TextCNN vs BiLSTM
Team: Pankhuri Kumari (230911084), Jambhorkar Arya Sachin (230911088), Sai Prasad Prusty (230911298)

This script is written to run in Google Colab against the REAL Amazon Polarity
dataset (via HuggingFace `datasets`). Set USE_REAL_DATA = True in Colab.

It was validated end-to-end in a sandboxed dev environment with no internet
access to huggingface.co, so USE_REAL_DATA = False here generates a small
synthetic stand-in dataset (templated positive/negative product-review
sentences) purely to prove the pipeline runs without bugs. The numbers
produced with USE_REAL_DATA=False are NOT meant for the report — they are a
smoke test. Re-run this notebook in Colab with USE_REAL_DATA=True to get the
real preliminary numbers for submission.
"""

import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

USE_REAL_DATA = False   # <-- set True in Colab
N_SAMPLES = 20000        # balanced subset size (10k pos + 10k neg), per synopsis
MAX_VOCAB = 20000
MAX_LEN = 100
EMBED_DIM = 100

# ---------------------------------------------------------------------------
# 1. Dataset acquisition
# ---------------------------------------------------------------------------

def load_real_amazon_polarity(n_samples=N_SAMPLES, seed=SEED):
    """Loads the Amazon Polarity dataset from HuggingFace and returns a
    balanced subset as a pandas DataFrame with columns ['text', 'label'].
    label: 1 = positive, 0 = negative. Run this in Colab (needs internet).
    """
    from datasets import load_dataset
    # The old bare "amazon_polarity" repo id was retired on the HF Hub;
    # it now lives under the fancyzhx namespace.
    ds = load_dataset("fancyzhx/amazon_polarity", split="train")
    df = ds.to_pandas()
    df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")
    pos = df[df.label == 1].sample(n=n_samples // 2, random_state=seed)
    neg = df[df.label == 0].sample(n=n_samples // 2, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out[["text", "label"]]


def load_synthetic_dev_data(n_samples=600, seed=SEED):
    """Small templated dataset used ONLY to smoke-test the pipeline offline.
    Not representative of Amazon Polarity — do not report these numbers.
    """
    rng = random.Random(seed)
    pos_templates = [
        "This {item} is absolutely fantastic, I love it and would buy again.",
        "Great {item}, works perfectly and the quality is excellent.",
        "I am very happy with this {item}, highly recommend to everyone.",
        "Amazing {item}! Exceeded my expectations, five stars.",
        "The {item} arrived on time and works flawlessly, superb value.",
        "Best {item} I have ever purchased, fast shipping and great build.",
    ]
    neg_templates = [
        "This {item} is terrible, it broke after one use, do not buy.",
        "Very disappointed with this {item}, poor quality and overpriced.",
        "The {item} stopped working within a week, waste of money.",
        "Awful {item}, not as described and customer service was rude.",
        "I regret buying this {item}, it is cheaply made and useless.",
        "Worst {item} ever, arrived damaged and packaging was bad.",
    ]
    items = ["phone case", "blender", "headphones", "backpack", "laptop stand",
             "coffee maker", "keyboard", "shoes", "watch", "charger",
             "monitor", "speaker", "vacuum", "camera", "toaster"]
    rows = []
    for _ in range(n_samples // 2):
        t = rng.choice(pos_templates).format(item=rng.choice(items))
        rows.append((t, 1))
        t = rng.choice(neg_templates).format(item=rng.choice(items))
        rows.append((t, 0))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z0-9!?',.\s]", " ", text)  # keep basic punctuation (negation cues)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_splits(df, seed=SEED):
    """80/10/10 train/val/test split, stratified on label."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df.label, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Model 1 — MLP on TF-IDF features  (owner: Pankhuri Kumari)
# ---------------------------------------------------------------------------

def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_mlp(train_df, val_df, test_df, epochs=5):
    vectorizer = TfidfVectorizer(max_features=MAX_VOCAB, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df.text).toarray()
    X_val = vectorizer.transform(val_df.text).toarray()
    X_test = vectorizer.transform(test_df.text).toarray()

    model = build_mlp(X_train.shape[1])
    model.fit(X_train, train_df.label.values,
              validation_data=(X_val, val_df.label.values),
              epochs=epochs, batch_size=32, verbose=0)

    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    return evaluate("MLP (TF-IDF)", test_df.label.values, preds)


# ---------------------------------------------------------------------------
# 4. Model 2 — TextCNN on word embeddings  (owner: Sai Prasad Prusty)
# ---------------------------------------------------------------------------

def build_textcnn(vocab_size):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_textcnn(tokenizer, X_train, X_val, X_test, train_df, val_df, test_df, epochs=5):
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    model = build_textcnn(vocab_size)
    model.fit(X_train, train_df.label.values,
              validation_data=(X_val, val_df.label.values),
              epochs=epochs, batch_size=32, verbose=0)
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    return evaluate("TextCNN", test_df.label.values, preds)


# ---------------------------------------------------------------------------
# 5. Model 3 — BiLSTM on word embeddings  (owner: Jambhorkar Arya Sachin)
# ---------------------------------------------------------------------------

def build_bilstm(vocab_size):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_bilstm(tokenizer, X_train, X_val, X_test, train_df, val_df, test_df, epochs=5):
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    model = build_bilstm(vocab_size)
    model.fit(X_train, train_df.label.values,
              validation_data=(X_val, val_df.label.values),
              epochs=epochs, batch_size=32, verbose=0)
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    return evaluate("BiLSTM", test_df.label.values, preds)


# ---------------------------------------------------------------------------
# 6. Shared evaluation
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    result = {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    print(f"{name:14s} | acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_real_amazon_polarity() if USE_REAL_DATA else load_synthetic_dev_data()
    df["text"] = df["text"].apply(clean_text)
    train_df, val_df, test_df = build_splits(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    results = []

    print("\nTraining MLP (TF-IDF baseline)...")
    results.append(run_mlp(train_df, val_df, test_df))

    print("\nTokenizing for CNN/BiLSTM (shared embedding pipeline)...")
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(train_df.text)
    X_train = pad_sequences(tok.texts_to_sequences(train_df.text), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val = pad_sequences(tok.texts_to_sequences(val_df.text), maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(tok.texts_to_sequences(test_df.text), maxlen=MAX_LEN, padding="post", truncating="post")

    print("\nTraining TextCNN...")
    results.append(run_textcnn(tok, X_train, X_val, X_test, train_df, val_df, test_df))

    print("\nTraining BiLSTM...")
    results.append(run_bilstm(tok, X_train, X_val, X_test, train_df, val_df, test_df))

    print("\n=== Comparison table (preliminary) ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv("preliminary_results.csv", index=False)
    return res_df


if __name__ == "__main__":
    main()
