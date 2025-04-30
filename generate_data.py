"""
Generate synthetic health data for Assignment 5.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_health_data(n_patients=150, records_per_patient_range=(30, 70), seed=42):
    """
    Generate synthetic health data for multiple patients.
    
    Args:
        n_patients: Number of patients to generate
        records_per_patient_range: Range of records per patient (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing synthetic health data
    """
    np.random.seed(seed)
    
    # Lists to store data
    all_data = []
    
    # Patient characteristics that remain constant
    patient_ages = np.random.normal(50, 15, n_patients).astype(int)
    patient_ages = np.clip(patient_ages, 18, 90)  # Clip to reasonable age range
    
    # Smoker status options
    smoker_options = ['no', 'yes', 'former']
    smoker_probs = [0.6, 0.2, 0.2]
    
    # Generate data for each patient
    for patient_id in range(1, n_patients + 1):
        # Number of records for this patient
        n_records = np.random.randint(records_per_patient_range[0], records_per_patient_range[1])
        
        # Patient's age
        age = patient_ages[patient_id - 1]
        
        # Patient's smoker status
        smoker_status = np.random.choice(smoker_options, p=smoker_probs)
        
        # Base values for this patient (with some randomness)
        base_systolic = 120 + np.random.normal(0, 10)
        base_diastolic = 80 + np.random.normal(0, 5)
        base_glucose = 100 + np.random.normal(0, 10)
        base_bmi = 25 + np.random.normal(0, 3)
        base_heart_rate = 70 + np.random.normal(0, 5)
        
        # Determine if patient has disease (more likely with age and if smoker)
        disease_prob = 0.05  # Base probability
        if age > 60:
            disease_prob += 0.05
        if smoker_status == 'yes':
            disease_prob += 0.1
        elif smoker_status == 'former':
            disease_prob += 0.03
        
        disease_outcome = 1 if np.random.random() < disease_prob else 0
        
        # If patient has disease, adjust base values
        if disease_outcome == 1:
            base_systolic += np.random.uniform(5, 15)
            base_diastolic += np.random.uniform(3, 10)
            base_glucose += np.random.uniform(10, 30)
            base_heart_rate += np.random.uniform(5, 15)
        
        # Generate records over time
        start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 30))
        
        for i in range(n_records):
            # Timestamp with increasing days
            timestamp = start_date + timedelta(days=i * np.random.uniform(1, 3))
            
            # Add random variations to base values
            systolic_bp = base_systolic + np.random.normal(0, 8)
            diastolic_bp = base_diastolic + np.random.normal(0, 5)
            glucose_level = base_glucose + np.random.normal(0, 15)
            bmi = base_bmi + np.random.normal(0, 0.5)
            heart_rate = base_heart_rate + np.random.normal(0, 8)
            
            # Ensure values are in reasonable ranges
            systolic_bp = max(90, min(200, systolic_bp))
            diastolic_bp = max(60, min(120, diastolic_bp))
            glucose_level = max(70, min(300, glucose_level))
            bmi = max(15, min(45, bmi))
            heart_rate = max(40, min(180, heart_rate))
            
            # Add occasional missing values
            if np.random.random() < 0.05:
                systolic_bp = np.nan
            if np.random.random() < 0.05:
                glucose_level = np.nan
            
            # Add record to dataset
            record = {
                'patient_id': patient_id,
                'timestamp': timestamp,
                'age': age,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'glucose_level': glucose_level,
                'bmi': bmi,
                'smoker_status': smoker_status,
                'heart_rate': heart_rate,
                'disease_outcome': disease_outcome
            }
            
            all_data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by patient_id and timestamp
    df = df.sort_values(['patient_id', 'timestamp'])
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_health_data()
    
    # Print summary
    print(f"\nGenerated {len(df)} records for {df['patient_id'].nunique()} patients.")
    
    # Show target distribution
    print("\nTarget variable distribution ('disease_outcome'):")
    print(df['disease_outcome'].value_counts(normalize=True).rename('proportion'))
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/synthetic_health_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSynthetic data saved to {output_file}")