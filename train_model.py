"""
Script to train and save the best-performing Random Survival Forest model.
Run this to generate the 'best_model_random_forest.pkl' file locally.
"""

import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import os

def main():
    print("⏳ Loading and preparing data...")
    # Load your cleaned data
    df = pd.read_csv('metabric_processed_clean.csv')
    
    # Define features and target
    os_covariates = [ 
        'Age at Diagnosis', 'Type of Breast Surgery', 'Cancer Type', 'Cancer Type Detailed',
        'Cellularity', 'Chemotherapy', 'Pam50 + Claudin-low subtype', 'Cohort',
        'ER status measured by IHC', 'ER Status', 'Neoplasm Histologic Grade',
        'HER2 status measured by SNP6', 'HER2 Status', 'Tumor Other Histologic Subtype',
        'Hormone Therapy', 'Inferred Menopausal State', 'Primary Tumor Laterality',
        'Lymph nodes examined positive', 'Mutation Count', 'Nottingham prognostic index',
        'Oncotree Code', 'PR Status', 'Radio Therapy', '3-Gene classifier subtype', 
        'Tumor Size', 'Tumor Stage'
    ]
    
    duration_col = 'Overall Survival (Months)'
    event_col = 'Overall Survival Status'
    
    # Create feature matrix X
    X = df[os_covariates].copy()
    
    # --- CRITICAL STEP: ENCODE CATEGORICAL VARIABLES ---
    print("Encoding categorical variables...")
    # Identify categorical columns (assuming object dtype)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding columns: {categorical_cols}")
    
    # Initialize a dictionary to store label encoders
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit and transform the column, ton handle any unseen categories safely
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    # --- END... ENCODING STEP ---
    
    # Create target structured array
    y_structured = Surv.from_arrays(
        event=(df[event_col] == 'Deceased').astype(bool),
        time=df[duration_col].values
    )
    
    # Split the data (use a fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_structured, test_size=0.2, random_state=42, stratify=df['Cancer Type Detailed']
    )
    
    print("Training Random Survival Forest model (This may take a few minutes)...")
    start_time = time.time()
    
    # Train the model with the EXACT same parameters
    best_model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1 
    )
    
    best_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Quick validation
    predictions = best_model.predict(X_test)
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(
        y_test['event'], 
        y_test['time'], 
        predictions
    )[0]
    
    print(f"Model trained successfully in {training_time:.2f} seconds!")
    print(f"Validation C-index: {c_index:.4f}")
    
    # Save the model
    model_filename = 'best_model_random_forest.pkl'
    joblib.dump(best_model, model_filename)
    
    # Also save the label encoders so the app can use them. So as to avoid conflicts
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print(f"Model saved as '{model_filename}'")
    print(f"Label encoders saved as 'label_encoders.pkl'")

    # ALSO save the feature names in the exact order used for training
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')  # Save the ordered list

    print(f"Feature names saved as 'feature_names.pkl'")

    # Calculate min/max from training predictions
    train_predictions = best_model.predict(X_train)
    min_score = np.min(train_predictions)
    max_score = np.max(train_predictions)

    # Save the normalization parameters
    normalization_params = {'min': min_score, 'max': max_score}
    joblib.dump(normalization_params, 'normalization_params.pkl')

        
    # Extra instruction for the Streamlit app to guide users
    print("\n" + "="*50)
    print("Next step: Run the Streamlit app with:")
    print("$ streamlit run streamlit_app.py") #ensure stramlit code name matches this
    print("="*50)

if __name__ == "__main__":
    main()
