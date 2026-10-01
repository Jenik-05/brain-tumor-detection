"""Builds the train / validation / test tables from the dataset manifest."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config


def create_splits(save=True):
    """Return (train_df, val_df, test_df).

    - Test    = the official Testing folder, untouched.
    - Train + validation = the cleaned Training images (exact duplicates and
      images that leak into Testing are excluded).
    - Near-duplicate Training images stay on the SAME side of the
      train/validation split, so validation is not inflated.
    """
    manifest = pd.read_csv(config.MANIFEST_PATH)

    # Label encoding: class name -> number (glioma=0, meningioma=1, ...)
    label_of = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    manifest["label"] = manifest["class"].map(label_of)

    # Full path of each image on this computer
    manifest["filepath"] = manifest["relative_path"].apply(
        lambda p: str(config.PROJECT_ROOT / p)
    )

    train_pool = manifest[
        (manifest["split"] == "Training") & (~manifest["exclude_from_training"])
    ].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "Testing"].reset_index(drop=True)

    # StratifiedGroupKFold: keeps class proportions AND keeps images with the
    # same fingerprint (dhash16) together. Taking one fold of 5 = 20% validation.
    n_folds = round(1 / config.VALIDATION_FRACTION)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=config.SEED)
    train_idx, val_idx = next(
        splitter.split(train_pool, train_pool["label"], groups=train_pool["dhash16"])
    )
    train_df = train_pool.iloc[train_idx].reset_index(drop=True)
    val_df = train_pool.iloc[val_idx].reset_index(drop=True)

    # Safety check: no fingerprint may appear on both sides
    overlap = set(train_df["dhash16"]) & set(val_df["dhash16"])
    assert len(overlap) == 0, f"{len(overlap)} fingerprints are in both train and validation!"

    if save:
        config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(config.SPLITS_DIR / "train.csv", index=False)
        val_df.to_csv(config.SPLITS_DIR / "validation.csv", index=False)
        test_df.to_csv(config.SPLITS_DIR / "test.csv", index=False)

    return train_df, val_df, test_df