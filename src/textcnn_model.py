"""
Model 2 — TextCNN: Conv1D over learned word embeddings.
Owner: Sai Prasad Prusty (230911298)
"""

from tensorflow.keras import layers, models

MAX_LEN = 100
EMBED_DIM = 100


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


def run_textcnn(vocab_size, X_train, X_val, X_test, train_df, val_df, test_df, evaluate_fn, epochs=5):
    model = build_textcnn(vocab_size)
    model.fit(X_train, train_df.label.values,
              validation_data=(X_val, val_df.label.values),
              epochs=epochs, batch_size=32, verbose=1)
    preds = (model.predict(X_test) > 0.5).astype(int).ravel()
    return evaluate_fn("TextCNN", test_df.label.values, preds)
