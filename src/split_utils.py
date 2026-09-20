"""
Stratified train/val/test split utilities.
Owner: Sai Prasad Prusty (230911298)
"""

from sklearn.model_selection import train_test_split

SEED = 42


def build_splits(df, seed=SEED):
    """80/10/10 train/val/test split, stratified on label, with a fixed
    seed so all three models (MLP, TextCNN, BiLSTM) see identical splits."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df.label, random_state=seed)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))
