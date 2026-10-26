"""Single-batch densification and a common validation-based stopping policy."""
import math

import numpy as np
from scipy import sparse


def dense_batches(features, labels=None, batch_size=32, shuffle=False, seed=42):
    order = np.arange(features.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        ids = order[start:start+batch_size]
        batch = features[ids]
        if sparse.issparse(batch):
            batch = batch.toarray().astype(np.float32, copy=False)
        else:
            batch = np.asarray(batch)
        yield batch if labels is None else (batch, np.asarray(labels[ids], dtype=np.float32))


def make_dataset(features, labels, batch_size=32, shuffle=False, seed=42):
    import tensorflow as tf
    epoch = [0]
    def generate():
        epoch_seed = seed + epoch[0]
        epoch[0] += 1
        yield from dense_batches(features, labels, batch_size, shuffle, epoch_seed)
    dtype = tf.float32 if sparse.issparse(features) else tf.as_dtype(features.dtype)
    dataset = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((None, features.shape[1]), dtype), tf.TensorSpec((None,), tf.float32)))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(math.ceil(features.shape[0]/batch_size)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.autotune.enabled = False
    options.deterministic = True
    return dataset.with_options(options).prefetch(1)


def fit_model(model, train, y_train, validation, y_validation, epochs=8, batch_size=32,
              patience=2, seed=42, checkpoint=None):
    import tensorflow as tf
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience,
                                                 restore_best_weights=True, verbose=1),
                 tf.keras.callbacks.TerminateOnNaN()]
    if checkpoint is not None:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True))
    history = model.fit(make_dataset(train, y_train, batch_size, True, seed),
                        validation_data=make_dataset(validation, y_validation, batch_size),
                        epochs=epochs, callbacks=callbacks, verbose=2)
    if not np.isfinite(np.asarray(history.history["val_loss"])).all():
        raise ValueError("Nonfinite validation loss")
    return history


def predict_batches(model, features, batch_size=32):
    outputs = [np.asarray(model(batch, training=False)).reshape(-1)
               for batch in dense_batches(features, batch_size=batch_size)]
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    return probabilities
