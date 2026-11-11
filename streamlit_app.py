"""OncoVision: breast cancer overall-survival risk groups (research prototype).

The app trains the same Random Survival Forest that run_analysis.py evaluated,
on the same training patients, so its risk groups are the ones whose survival
is reported in results/risk_groups.csv.
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

import oncovision as ov

st.set_page_config(page_title="OncoVision", page_icon="🧬", layout="wide")
GROUPS = ["Low", "Medium", "High"]
COLORS = {"Low": "#2a7f86", "Medium": "#c98a12", "High": "#c0392b"}


@st.cache_resource(show_spinner="Training the model on 1,584 patients (first visit only)...")
def load():
    df = pd.read_csv(ov.CLEAN_FILE)
    X, y = df[ov.FEATURES], ov.target(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y["event"], random_state=ov.RANDOM_STATE)
    model = ov.models(low_memory=True)["Random Survival Forest"]
    model.set_params(model__oob_score=True).fit(X_train, y_train)
    # Out-of-bag scores: each training patient scored by trees that never saw them.
    oob = model.named_steps["model"].oob_prediction_
    cuts = np.quantile(oob, [1 / 3, 2 / 3])
    return model, np.sort(oob), cuts, X_train


model, oob_sorted, cuts, X_train = load()
results = Path("results")
group_stats = pd.read_csv(results / "risk_groups.csv").set_index("risk_group")
bench = pd.read_csv(results / "benchmark.csv")


def figure(container, name, caption):
    # Embedded as data URIs: Streamlit Cloud's media endpoint returned HTTP 500
    # for st.image files, so the figures are sent inline with the page instead.
    data = base64.b64encode((results / "figures" / name).read_bytes()).decode()
    container.markdown(
        f"<figure style='margin:0'><img src='data:image/png;base64,{data}' style='width:100%'>"
        f"<figcaption style='font-size:0.85rem;color:gray'>{caption}</figcaption></figure>",
        unsafe_allow_html=True)

st.title("OncoVision: breast cancer survival risk groups")
st.caption("Research prototype built on the METABRIC cohort. Not a medical device and "
           "not for treatment decisions.")

tab_predict, tab_model = st.tabs(["Assess a patient", "How the model was built"])

with tab_predict:
    st.write("Enter what is known. Leave anything unknown blank or as **Unknown**; "
             "it is filled in the same way missing values were filled during training.")
    with st.form("patient"):
        values = {}
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Tumour")
            values["Age at Diagnosis"] = st.number_input("Age at diagnosis (years)", 20.0, 100.0, 60.0)
            values["Nottingham prognostic index"] = st.number_input("Nottingham prognostic index", 1.0, 7.0, 4.0, 0.1)
            values["Lymph nodes examined positive"] = st.number_input("Positive lymph nodes", 0, 50, 0)
            values["Tumor Size"] = st.number_input("Tumour size (mm)", 1.0, 200.0, 25.0)
            values["Tumor Stage"] = st.number_input("Tumour stage (0-4)", 0, 4, value=None, placeholder="Unknown")
            values["Neoplasm Histologic Grade"] = st.number_input("Histologic grade (1-3)", 1, 3, value=None, placeholder="Unknown")
            values["Mutation Count"] = st.number_input("Mutation count", 0, 100, value=None, placeholder="Unknown")

        def pick(label, col):
            opts = ["Unknown"] + sorted(X_train[col].dropna().unique().tolist())
            choice = st.selectbox(label, opts)
            values[col] = np.nan if choice == "Unknown" else choice

        with c2:
            st.subheader("Biology")
            pick("Cancer type", "Cancer Type Detailed")
            pick("Cellularity", "Cellularity")
            pick("PAM50 + claudin-low subtype", "Pam50 + Claudin-low subtype")
            pick("ER status", "ER Status")
            pick("PR status", "PR Status")
            pick("HER2 status", "HER2 Status")
            pick("HER2 copy number (SNP6)", "HER2 status measured by SNP6")
            pick("3-gene classifier subtype", "3-Gene classifier subtype")
        with c3:
            st.subheader("Patient and treatment")
            pick("Menopausal state", "Inferred Menopausal State")
            pick("Tumour laterality", "Primary Tumor Laterality")
            pick("Breast surgery", "Type of Breast Surgery")
            pick("Chemotherapy", "Chemotherapy")
            pick("Hormone therapy", "Hormone Therapy")
            pick("Radiotherapy", "Radio Therapy")
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        row = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in values.items()}])[ov.FEATURES]
        score = float(model.predict(row)[0])
        group = GROUPS[int(np.digitize(score, cuts))]
        pct = 100 * np.searchsorted(oob_sorted, score) / len(oob_sorted)
        s = group_stats.loc[group]

        st.markdown(f"### Risk group: <span style='color:{COLORS[group]}'>{group}</span>",
                    unsafe_allow_html=True)
        st.write(f"Higher predicted risk than **{pct:.0f}%** of the training patients.")
        m1, m2, m3 = st.columns(3)
        m1.metric("5-year survival in this group", f"{s.survival_5y:.0%}")
        m2.metric("10-year survival in this group", f"{s.survival_10y:.0%}")
        m3.metric("Died during follow-up", f"{s.event_rate:.0%}")
        st.caption(f"Group figures come from {int(s.patients)} held-out test patients in the same "
                   "group (Kaplan-Meier estimates). They describe the group, not this patient.")

with tab_model:
    st.subheader("Data")
    st.write("METABRIC: 2,509 patients in the source file. 528 had no recorded survival outcome "
             "and were removed rather than guessed, leaving **1,981 patients** (1,144 deaths). "
             "21 clinical features. 80% of patients (1,584) were used for training and 20% (397) "
             "were held back for testing. Missing feature values were filled in inside the model, "
             "using training patients only.")
    st.subheader("Benchmark")
    show = bench.rename(columns={"model": "Model", "cv_c_index_mean": "CV C-index",
                                 "cv_c_index_sd": "CV SD", "test_c_index": "Test C-index",
                                 "test_auc_5y": "Test AUC 5y", "test_auc_10y": "Test AUC 10y"})
    st.dataframe(show[["Model", "CV C-index", "CV SD", "Test C-index", "Test AUC 5y", "Test AUC 10y"]],
                 hide_index=True)
    st.write("The model was chosen by 5-fold cross-validation on the training patients. "
             "Random Survival Forest and Gradient Boosting were effectively tied.")
    figure(st, "risk_groups_km.png", "Survival of held-out test patients by risk group")
    c1, c2 = st.columns(2)
    figure(c1, "permutation_importance.png", "Permutation importance (drop in test C-index)")
    figure(c2, "shap_beeswarm.png", "SHAP values for 200 test patients")
    st.info("Treatment variables (surgery, radiotherapy, chemotherapy) reflect who received them, "
            "not their effect: sicker patients get more aggressive treatment. Do not read them as "
            "treatment advice.")
