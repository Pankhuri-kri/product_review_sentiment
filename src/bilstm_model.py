"""
Model 3 — BiLSTM: reads the review sequentially in both directions to
capture word order and longer-range context.
Owner: Jambhorkar Arya Sachin (230911088)
"""

from tensorflow.keras import layers, models, optimizers

MAX_LEN = 100
EMBED_DIM = 100


def build_bilstm(vocab_size, dropout=0.3, learning_rate=0.001, weight_decay=0.0001):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,), dtype="int32"),
        layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_bilstm(vocab_size, X_train, X_val, X_test, train_df, val_df, test_df, evaluate_fn, epochs=5):
    model = build_bilstm(vocab_size)
    from training_utils import fit_model, predict_batches
    fit_model(model, X_train, train_df.label.values, X_val, val_df.label.values, epochs=epochs)
    preds = (predict_batches(model, X_test) > 0.5).astype(int)
    return evaluate_fn("BiLSTM", test_df.label.values, preds)
