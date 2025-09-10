#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="OncoVision: Breast Cancer Risk Predictor",
    page_icon="🧬", 
    layout="wide"
)

# Load resources
@st.cache_resource
def load_feature_names():
    try:
        return joblib.load('models/feature_names.pkl')
    except:
        st.error("Feature names not found. Run train_model.py first.")
        return None

def load_encoders():
    try:
        return joblib.load('models/label_encoders.pkl')
    except:
        st.error("Encoders not found. Run train_model.py first.")
        return None

def load_model():
    try:
        return joblib.load('models/best_model_random_forest.pkl')
    except:
        st.error("Model not found. Run train_model.py first.")
        return None

def load_normalization_params():
    try:
        return joblib.load('models/normalization_params.pkl')
    except:
        # Fallback to reasonable defaults
        return {'min': 0, 'max': 300}

# App title
st.title("🧬 Breast Cancer Survival Risk Predictor")
st.write("Enter patient clinical details to predict survival risk")

# Create form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age at Diagnosis (years)", 25.0, 95.0, 60.0)
        nottingham_index = st.slider("Nottingham Prognostic Index", 2.0, 8.0, 4.0)
        lymph_nodes = st.slider("Positive Lymph Nodes", 0, 30, 2)
        tumor_size = st.slider("Tumor Size (mm)", 5.0, 100.0, 25.0)
        tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4])
        
    with col2:
        surgery_type = st.selectbox("Surgical Approach", ["Mastectomy", "Breast Conserving"])
        cancer_type = st.selectbox("Cancer Type", ["Breast Cancer", "Breast Sarcoma"])
        cellularity = st.selectbox("Cellularity", ["High", "Moderate", "Low"])
        chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        er_status = st.selectbox("ER Status", ["Positive", "Negative"])
        her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
    
    # Hidden defaults for other features... added to prevent issues with missing values passed in durinf fit
    defaults = {
        'Cancer Type Detailed': 'Invasive Ductal Carcinoma',
        'Pam50 + Claudin-low subtype': 'Luminal A',
        'Cohort': 1.0,
        'ER status measured by IHC': 'Positive',
        'Neoplasm Histologic Grade': 2,
        'HER2 status measured by SNP6': 'Neutral',
        'Tumor Other Histologic Subtype': 'Ductal/NST',
        'Hormone Therapy': 'Yes',
        'Inferred Menopausal State': 'Post',
        'Primary Tumor Laterality': 'Right',
        'Mutation Count': 2,
        'Oncotree Code': 'IDC',
        'PR Status': 'Positive',
        'Radio Therapy': 'Yes',
        '3-Gene classifier subtype': 'ER+/HER2- High Prolif'
    }
    
    submitted = st.form_submit_button("Predict Risk")

# Process prediction
if submitted:
    # Load required resources
    feature_names = load_feature_names()
    encoders = load_encoders()
    model = load_model()
    
    if None in [feature_names, encoders, model]:
        st.stop()
    
    # Prepare input data
    input_values = {
        'Age at Diagnosis': age,
        'Nottingham prognostic index': nottingham_index,
        'Lymph nodes examined positive': lymph_nodes,
        'Type of Breast Surgery': surgery_type,
        'Tumor Size': tumor_size,
        'Tumor Stage': tumor_stage,
        'Cancer Type': cancer_type,
        'Cellularity': cellularity,
        'Chemotherapy': chemotherapy,
        'ER Status': er_status,
        'HER2 Status': her2_status,
        **defaults  # Add all the default values
    }
    
    # Create DataFrame with correct column order 
    input_df = pd.DataFrame(columns=feature_names)
    for feature in feature_names:
        if feature in input_values:
            input_df[feature] = [input_values[feature]]
        else:
            st.error(f"Missing value for feature: {feature}")
            st.stop()
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Make prediction
    try:
        risk_score = model.predict(input_df)[0]
        
        # --- NORMALIZE THE RISK SCORE ---
        params = load_normalization_params()
        normalized_score = (risk_score - params['min']) / (params['max'] - params['min'])
        normalized_score = max(0, min(1, normalized_score))  # Clamp to 0-1 range
        # --- END NORMALIZATION ---
        
        # Determine risk group using normalized score
        if normalized_score < 0.25:
            risk_group = "Low Risk"
            color = "green"
            interpretation = "Favorable prognosis"
        elif normalized_score < 0.65:
            risk_group = "Intermediate Risk"
            color = "orange"
            interpretation = "Moderate prognosis"
        else:
            risk_group = "High Risk"
            color = "red"
            interpretation = "Poor prognosis"
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw Risk Score", f"{risk_score:.1f}")
        with col2:
            st.metric("Normalized Risk", f"{normalized_score:.3f}")
        
        st.markdown(f"<h3 style='color: {color}'>Risk Category: {risk_group}</h3>", 
                   unsafe_allow_html=True)
        st.write(f"**Interpretation:** {interpretation}")
        
        # Visual risk meter
        st.progress(float(normalized_score))
        st.caption(f"Risk level: {normalized_score:.1%}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Sidebar info
with st.sidebar:
    st.info("""
    **How to use:**
    1. Fill in the patient details
    2. Click 'Predict Risk'
    3. View the risk assessment
    
    **Note:** This is for research purposes only.
    """)
