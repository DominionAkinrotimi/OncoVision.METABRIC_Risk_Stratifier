"""Shared data cleaning and model code for OncoVision.

Both the analysis script (run_analysis.py) and the Streamlit app import from
here, so the app scores patients with exactly the pipeline that was evaluated.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

RAW_FILE = "Breast Cancer METABRIC.csv"
CLEAN_FILE = "data/metabric_clean.csv"

TIME_COL = "Overall Survival (Months)"
EVENT_COL = "Overall Survival Status"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age at Diagnosis",
    "Nottingham prognostic index",
    "Lymph nodes examined positive",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "Mutation Count",
]

CATEGORICAL_FEATURES = [
    "Type of Breast Surgery",
    "Cancer Type Detailed",
    "Cellularity",
    "Chemotherapy",
    "Hormone Therapy",
    "Radio Therapy",
    "Pam50 + Claudin-low subtype",
    "ER Status",
    "PR Status",
    "HER2 Status",
    "HER2 status measured by SNP6",
    "3-Gene classifier subtype",
    "Inferred Menopausal State",
    "Primary Tumor Laterality",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns left out of the model, and why. Printed by run_analysis.py.
EXCLUDED = {
    "Patient ID": "identifier",
    "Sex": "every patient is female",
    "Cancer Type": "3 sarcomas out of 1,981; no usable signal",
    "Cohort": "recruitment batch, not a patient characteristic",
    "ER status measured by IHC": "duplicates ER Status",
    "Oncotree Code": "duplicates Cancer Type Detailed",
    "Tumor Other Histologic Subtype": "duplicates Cancer Type Detailed",
    "Integrative Cluster": "research-only genomic label, not available in clinic",
    "Relapse Free Status": "outcome measured after diagnosis (leakage)",
    "Relapse Free Status (Months)": "outcome measured after diagnosis (leakage)",
    "Patient's Vital Status": "restates the outcome (leakage)",
}


def load_raw(path=RAW_FILE):
    return pd.read_csv(path)


def clean(raw):
    """Return (clean dataframe, report dict).

    Patients with no recorded survival outcome are dropped, never imputed:
    imputing the label would train and score the model on invented outcomes.
    Missing feature values are left as NaN here and imputed inside the model
    pipeline, so imputation is learned from training data only.
    """
    report = {"raw_rows": len(raw), "raw_columns": raw.shape[1],
              "raw_missing_values": int(raw.isna().sum().sum())}

    df = raw[raw[EVENT_COL].notna() & raw[TIME_COL].notna()].copy()
    report["dropped_no_outcome"] = report["raw_rows"] - len(df)

    # Fix spelling so values are consistent.
    df["ER status measured by IHC"] = df["ER status measured by IHC"].replace("Positve", "Positive")
    # 'Undef' is the source file's own missing marker.
    df["HER2 status measured by SNP6"] = df["HER2 status measured by SNP6"].replace("Undef", np.nan)

    df["event"] = df[EVENT_COL].eq("Deceased")
    df["time"] = df[TIME_COL].astype(float)

    df = df[["Patient ID"] + FEATURES + ["time", "event"]].reset_index(drop=True)
    report["patients"] = len(df)
    report["features"] = len(FEATURES)
    report["events"] = int(df["event"].sum())
    report["censored"] = int((~df["event"]).sum())
    report["missing_feature_values"] = int(df[FEATURES].isna().sum().sum())
    report["missing_by_feature"] = {k: int(v) for k, v in df[FEATURES].isna().sum().items() if v}
    return df, report


def target(df):
    return Surv.from_arrays(event=df["event"].to_numpy(), time=df["time"].to_numpy())


def preprocessor(scale):
    numeric_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        numeric_steps.append(StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # drop='first' keeps the Cox design matrix full rank.
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [("num", make_pipeline(*numeric_steps), NUMERIC_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def models(low_memory=False):
    """The four benchmarked models. Hyperparameters are fixed in advance
    (common defaults), not tuned on the test set."""
    return {
        "CoxPH": Pipeline([
            ("prep", preprocessor(scale=True)),
            # A tiny ridge penalty only to keep the fit numerically stable.
            ("model", CoxPHSurvivalAnalysis(alpha=0.01, ties="efron")),
        ]),
        "CoxNet (elastic net)": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("model", CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                             fit_baseline_model=True)),
        ]),
        "Random Survival Forest": Pipeline([
            ("prep", preprocessor(scale=False)),
            ("model", RandomSurvivalForest(n_estimators=300, min_samples_split=10,
                                           min_samples_leaf=15, max_features="sqrt",
                                           n_jobs=-1, random_state=RANDOM_STATE,
                                           low_memory=low_memory)),
        ]),
        "Gradient Boosting Survival": Pipeline([
            ("prep", preprocessor(scale=False)),
            ("model", GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=RANDOM_STATE)),
        ]),
    }


def tune_coxnet(X_train, y_train, cv):
    """CoxNet's penalty strength (alpha) has no sensible default, so it is
    chosen by 5-fold CV on the training set only."""
    from sklearn.model_selection import GridSearchCV

    path = models()["CoxNet (elastic net)"]
    path.set_params(model__n_alphas=30).fit(X_train, y_train)
    alphas = path.named_steps["model"].alphas_
    grid = GridSearchCV(
        models()["CoxNet (elastic net)"],
        {"model__alphas": [[a] for a in alphas]},
        cv=cv, n_jobs=-1, error_score=0.5,
    ).fit(X_train, y_train)
    return grid.best_params_["model__alphas"][0]
