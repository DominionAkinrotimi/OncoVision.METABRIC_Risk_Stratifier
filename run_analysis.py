"""Clean the data, benchmark four survival models and explain the best one.

Run:  python run_analysis.py
Writes data/metabric_clean.csv and everything in results/.
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sksurv.compare import compare_survival
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.nonparametric import kaplan_meier_estimator

import oncovision as ov

warnings.filterwarnings("ignore", category=FutureWarning)
# Expected and harmless: the largest penalties in the CoxNet grid zero every
# coefficient, and a few rare categories are absent from some CV folds.
warnings.filterwarnings("ignore", message="all coefficients are zero")
warnings.filterwarnings("ignore", message="Found unknown categories")
OUT = Path("results")
FIG = OUT / "figures"
TAU = 240  # months; Uno's C-index is computed up to 20 years


def harrell(y, risk):
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # 1. Clean -----------------------------------------------------------
    df, report = ov.clean(ov.load_raw())
    df.to_csv(ov.CLEAN_FILE, index=False)
    report["excluded_columns"] = ov.EXCLUDED
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_columns"}, indent=2))

    X, y = df[ov.FEATURES], ov.target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y["event"], random_state=ov.RANDOM_STATE)
    report["train_patients"], report["test_patients"] = len(X_train), len(X_test)
    (OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    cv = list(StratifiedKFold(5, shuffle=True, random_state=ov.RANDOM_STATE)
              .split(X_train, y_train["event"]))

    # 2. Benchmark --------------------------------------------------------
    candidates = ov.models(low_memory=True)
    alpha = ov.tune_coxnet(X_train, y_train, cv)
    candidates["CoxNet (elastic net)"].set_params(model__alphas=[alpha])
    print(f"CoxNet alpha chosen by CV: {alpha:.4g}")

    rows, fitted = [], {}
    for name, pipe in candidates.items():
        fold_scores = []
        for tr, va in cv:
            m = clone(pipe).fit(X_train.iloc[tr], y_train[tr])
            fold_scores.append(harrell(y_train[va], m.predict(X_train.iloc[va])))
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        risk = pipe.predict(X_test)
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, [60, 120])
        rows.append({
            "model": name,
            "cv_c_index_mean": np.mean(fold_scores),
            "cv_c_index_sd": np.std(fold_scores),
            "test_c_index": harrell(y_test, risk),
            "test_uno_c_index": concordance_index_ipcw(y_train, y_test, risk, tau=TAU)[0],
            "test_auc_5y": auc[0],
            "test_auc_10y": auc[1],
            "fit_seconds": fit_s,
        })
        fitted[name] = pipe
        print(f"{name:28s} CV {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}  "
              f"test {rows[-1]['test_c_index']:.4f}")

    bench = pd.DataFrame(rows).sort_values("cv_c_index_mean", ascending=False)
    bench.round(4).to_csv(OUT / "benchmark.csv", index=False)
    # The winner is picked on cross-validation, never on the test set.
    best_name = bench.iloc[0]["model"]
    best = fitted[best_name]
    print(f"Best by CV: {best_name}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    b = bench.iloc[::-1]
    ax.barh(b["model"], b["cv_c_index_mean"], xerr=b["cv_c_index_sd"], color="#2a7f86")
    ax.scatter(b["test_c_index"], b["model"], color="black", zorder=3, label="held-out test")
    ax.set_xlim(0.6, 0.75)
    ax.set_xlabel("C-index (bars: 5-fold CV mean ± SD)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "benchmark.png", dpi=150); plt.close(fig)

    # 3. Risk groups ------------------------------------------------------
    # Cut-offs are tertiles of training patients' out-of-bag scores: each
    # patient is scored only by trees that never saw them, so the cut-offs are
    # neither overfitted nor set using test patients. (A forest's score scale
    # depends on its training size, so cut-offs from smaller CV-fold models
    # would not transfer.) Other models fall back to out-of-fold scores.
    if best_name == "Random Survival Forest":
        best.set_params(model__oob_score=True).fit(X_train, y_train)
        honest_scores = best.named_steps["model"].oob_prediction_
    else:
        honest_scores = cross_val_predict(clone(best), X_train, y_train, cv=cv)
    cuts = np.quantile(honest_scores, [1 / 3, 2 / 3]).tolist()
    labels = np.array(["Low", "Medium", "High"])
    test_group = labels[np.digitize(best.predict(X_test), cuts)]

    groups, fig_ax = [], plt.subplots(figsize=(7, 4.5))
    fig, ax = fig_ax
    colors = {"Low": "#2a7f86", "Medium": "#e0a030", "High": "#c0392b"}
    for g in labels:
        yy = y_test[test_group == g]
        t, s = kaplan_meier_estimator(yy["event"], yy["time"])
        below = t[s <= 0.5]
        groups.append({"risk_group": g, "patients": len(yy),
                       "deaths": int(yy["event"].sum()),
                       "event_rate": yy["event"].mean(),
                       "median_survival_months": float(below[0]) if len(below) else None,
                       "survival_5y": float(s[t <= 60][-1]) if (t <= 60).any() else 1.0,
                       "survival_10y": float(s[t <= 120][-1]) if (t <= 120).any() else 1.0})
        ax.step(t, s, where="post", color=colors[g], label=f"{g} (n={len(yy)})")
    chi2, p = compare_survival(y_test, test_group)
    ax.set_xlabel("Months since diagnosis"); ax.set_ylabel("Overall survival")
    ax.set_title(f"Held-out test patients by risk group (log-rank p = {p:.1e})")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "risk_groups_km.png", dpi=150)
    plt.close(fig)
    groups = pd.DataFrame(groups)
    groups.round(4).to_csv(OUT / "risk_groups.csv", index=False)
    print(groups.round(3).to_string(index=False), f"\nlog-rank p = {p:.2e}")

    # 4. Permutation importance (raw features, held-out test set) ----------
    perm = permutation_importance(best, X_test, y_test, n_repeats=20,
                                  random_state=ov.RANDOM_STATE, n_jobs=-1)
    pi = (pd.DataFrame({"feature": ov.FEATURES, "importance_mean": perm.importances_mean,
                        "importance_sd": perm.importances_std})
          .sort_values("importance_mean", ascending=False))
    pi.round(4).to_csv(OUT / "permutation_importance.csv", index=False)
    top = pi.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#2a7f86")
    ax.set_xlabel("Drop in test C-index when the feature is shuffled")
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=150); plt.close(fig)

    # 5. SHAP -----------------------------------------------------------------
    import shap
    prep, model = best.named_steps["prep"], best.named_steps["model"]
    names = prep.get_feature_names_out()
    Xt_train = pd.DataFrame(prep.transform(X_train), columns=names)
    Xt_test = pd.DataFrame(prep.transform(X_test), columns=names)
    background = shap.sample(Xt_train, 50, random_state=ov.RANDOM_STATE)
    explain_rows = Xt_test.sample(min(200, len(Xt_test)), random_state=ov.RANDOM_STATE)
    explainer = shap.PermutationExplainer(lambda a: model.predict(np.asarray(a)), background, seed=ov.RANDOM_STATE)
    sv = explainer(explain_rows, max_evals=2 * len(names) + 1)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm.png", dpi=150); plt.close("all")

    # SHAP values are additive, so one-hot columns are summed back to the
    # original clinical variable before ranking.
    def source(col):
        for f in sorted(ov.FEATURES, key=len, reverse=True):
            if col == f or col.startswith(f + "_") or col == f"missingindicator_{f}":
                return f
        return col
    per_feature = pd.DataFrame(sv.values, columns=names).T.groupby(source).sum().T
    shap_rank = (per_feature.abs().mean().sort_values(ascending=False)
                 .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    shap_rank.round(4).to_csv(OUT / "shap_importance.csv", index=False)
    print("Top SHAP:", shap_rank.head(5).to_string(index=False))

    # 6. Settings the app needs -------------------------------------------
    (OUT / "app_config.json").write_text(json.dumps({
        "model": best_name, "coxnet_alpha": alpha, "risk_cutoffs": cuts,
        "train_patients": len(X_train), "test_patients": len(X_test),
        "log_rank_p": p,
    }, indent=2))


if __name__ == "__main__":
    main()
