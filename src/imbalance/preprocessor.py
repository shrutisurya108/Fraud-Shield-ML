"""
fraud-shield-ml | src/imbalance/preprocessor.py
─────────────────────────────────────────────────
Shared sklearn ColumnTransformer pipeline.

Handles:
  - Numeric columns  : median imputation → optional StandardScaler
  - Categorical cols : most-frequent imputation → OrdinalEncoder

Column type detection uses df.select_dtypes(exclude='number') which is
pandas' own internal API — completely immune to StringDtype, ArrowDtype,
and any pandas 2.x future.infer_string=True behaviour changes.

All categorical columns are cast to plain 'object' dtype before being
passed to sklearn, ensuring compatibility with all sklearn versions.

Used by:
  - Phase 4 imbalance comparison (Logistic Regression base)
  - Phase 5 model training (XGBoost, LightGBM, LR baseline)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# Columns to always drop before modelling — never features
DROP_COLS = {
    "TransactionID",
    "TransactionDT",
    "TransactionDateTime",
    "isFraud",
}


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numeric and categorical feature columns.

    Uses df.select_dtypes() — pandas' own internal method, fully immune to
    StringDtype / ArrowDtype / future.infer_string=True variations across
    pandas 2.x versions.

    Returns
    -------
    numeric_cols : list of numeric feature column names
    cat_cols     : list of non-numeric (categorical) feature column names
    """
    # Work only on available feature columns
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    df_features  = df[feature_cols]

    cat_cols     = df_features.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df_features.select_dtypes(include="number").columns.tolist()

    return numeric_cols, cat_cols


def build_preprocessor(
    numeric_cols: list[str],
    cat_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer.

    Parameters
    ----------
    numeric_cols  : list of numeric column names
    cat_cols      : list of categorical column names
    scale_numeric : if True, apply StandardScaler after imputation
                    (needed for Logistic Regression, not for tree models)
    """
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if cat_cols:
        transformers.append(("categorical", categorical_pipeline, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_xy(df: pd.DataFrame, scale_numeric: bool = True) -> tuple:
    """
    Split df into X (features) and y (target).

    Categorical columns are explicitly cast to plain object dtype before
    returning, ensuring sklearn compatibility regardless of pandas version.

    Returns
    -------
    X            : pd.DataFrame of features (unprocessed, cats as object)
    y            : pd.Series of integer labels
    numeric_cols : list
    cat_cols     : list
    """
    y = df["isFraud"].astype(int)
    numeric_cols, cat_cols = get_feature_columns(df)

    X = df[numeric_cols + cat_cols].copy()

    # Cast all categorical columns to plain object dtype.
    # This neutralises pandas 2.3 StringDtype / ArrowDtype so sklearn
    # SimpleImputer and OrdinalEncoder receive standard object arrays.
    for col in cat_cols:
        X[col] = X[col].astype(object)

    return X, y, numeric_cols, cat_cols
