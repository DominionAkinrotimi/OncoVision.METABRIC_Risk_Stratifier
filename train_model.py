# train_model.py
"""
Script to train and save the best-performing Random Survival Forest model.
Run this to generate the 'best_model_random_forest.pkl' file locally.
"""

import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import joblib
import time

def main():
    print("⏳ Loading and preparing data...")
    # Load your cleaned data
    df = pd.read_csv('metabric_processed_clean.csv')
    
    # Define your features and target (USE THE SAME FINAL LIST AS YOUR NOTEBOOK)
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
    
    X = df[os_covariates].copy()
    y_structured = Surv.from_arrays(
        event=(df[event_col] == 'Deceased').astype(bool),
        time=df[duration_col].values
    )
    
    # Split the data (use a fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_structured, test_size=0.2, random_state=42, stratify=df['Cancer Type Detailed']
    )
    
    print("🏗️ Training Random Survival Forest model (This may take a few minutes)...")
    start_time = time.time()
    
    # Train the model with the EXACT same parameters
    best_model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all available cores
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
    
    print(f"✅ Model trained successfully in {training_time:.2f} seconds!")
    print(f"✅ Validation C-index: {c_index:.4f}")
    
    # Save the model
    model_filename = 'best_model_random_forest.pkl'
    joblib.dump(best_model, model_filename)
    print(f"💾 Model saved as '{model_filename}'")
    
    # Print instructions for the Streamlit app
    print("\n" + "="*50)
    print("Next step: Run the Streamlit app with:")
    print("$ streamlit run streamlit_app.py")
    print("="*50)

if __name__ == "__main__":
    main()
