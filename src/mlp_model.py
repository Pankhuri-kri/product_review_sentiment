"""TF-IDF MLP baseline. Owner: Pankhuri Kumari (230911084)."""
from tensorflow.keras import layers, models, optimizers


def build_mlp(input_dim, dropout=0.3, learning_rate=0.001, weight_decay=0.0001):
    model = models.Sequential([
        layers.Input(shape=(input_dim,), dtype="float32"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp")
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
