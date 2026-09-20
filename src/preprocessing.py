"""
Text cleaning + shared tokenizer/embedding pipeline for TextCNN and BiLSTM.
Owner: Jambhorkar Arya Sachin (230911088)
"""

import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB = 20000
MAX_LEN = 100


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs, keep negation/emphasis punctuation (!, ?, ', ,, .)."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9!?',.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tokenizer(train_texts, max_vocab=MAX_VOCAB):
    """Fit a Keras Tokenizer on the training text only (never on val/test,
    to avoid leaking vocabulary information across the split)."""
    tok = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    return tok


def texts_to_padded_sequences(tokenizer, texts, max_len=MAX_LEN):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def vocab_size_of(tokenizer, max_vocab=MAX_VOCAB):
    return min(max_vocab, len(tokenizer.word_index) + 1)
