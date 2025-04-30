import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# --- Configuration ---
N_PATIENTS = 150
RECORDS_PER_PATIENT_AVG = 50
RECORDS_PER_PATIENT_STD = 15
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'synthetic_health_data.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Patient Demographics ---
np.random.seed(42)
patient_ids = [f'PAT_{i:03d}' for i in range(N_PATIENTS)]
ages = np.random.normal(loc=55, scale=15, size=N_PATIENTS).clip(18, 90).astype(int)
baseline_bmi = np.random.normal(loc=28, scale=5, size=N_PATIENTS).clip(18, 50)
baseline_systolic = np.random.normal(loc=125, scale=15, size=N_PATIENTS) + (baseline_bmi - 28) * 0.5 # BMI effect
baseline_diastolic = baseline_systolic - np.random.normal(loc=45, scale=5, size=N_PATIENTS) # Ensure diastolic < systolic
baseline_glucose = np.random.normal(loc=100, scale=20, size=N_PATIENTS) + (baseline_bmi - 28) * 0.3 # BMI effect
smoker_options = ['no', 'former', 'yes']
smoker_probs = [0.6, 0.25, 0.15]
smoker_status = np.random.choice(smoker_options, size=N_PATIENTS, p=smoker_probs)

patient_data = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'baseline_bmi': baseline_bmi,
    'baseline_systolic': baseline_systolic,
    'baseline_diastolic': baseline_diastolic,
    'baseline_glucose': baseline_glucose,
    'smoker_status': smoker_status
})

# --- Time Series Data Generation ---
all_records = []
total_days = (END_DATE - START_DATE).days

for i, patient in patient_data.iterrows():
    n_records = int(np.random.normal(loc=RECORDS_PER_PATIENT_AVG, scale=RECORDS_PER_PATIENT_STD))
    n_records = max(5, n_records) # Ensure at least a few records

    # Generate random timestamps within the date range
    random_days = np.random.randint(0, total_days, size=n_records)
    random_seconds = np.random.randint(0, 24*60*60, size=n_records)
    timestamps = [START_DATE + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]
    timestamps.sort()

    # Simulate physiological data with some noise and drift
    time_factor = np.linspace(0, 1, n_records) # Simple time progression factor
    hr = np.random.normal(loc=75, scale=8, size=n_records) + np.sin(time_factor * np.pi) * 5 # Some variation
    systolic = patient['baseline_systolic'] + np.random.normal(0, 5, n_records) + (time_factor - 0.5) * 3
    diastolic = patient['baseline_diastolic'] + np.random.normal(0, 4, n_records) + (time_factor - 0.5) * 2
    glucose = patient['baseline_glucose'] + np.random.normal(0, 5, n_records) + (time_factor - 0.5) * 2
    bmi = patient['baseline_bmi'] + np.random.normal(0, 0.5, n_records) # Less variation day-to-day

    patient_records = pd.DataFrame({
        'patient_id': patient['patient_id'],
        'timestamp': timestamps,
        'age': patient['age'], # Age is constant per patient for simplicity here
        'systolic_bp': systolic.clip(80, 200),
        'diastolic_bp': diastolic.clip(50, 130),
        'glucose_level': glucose.clip(50, 300),
        'bmi': bmi.clip(18, 50),
        'smoker_status': patient['smoker_status'],
        'heart_rate': hr.clip(40, 180)
    })
    all_records.append(patient_records)

final_df = pd.concat(all_records, ignore_index=True)

# --- Generate Imbalanced Target Variable ---
# Define risk factors
risk_score = (
    (final_df['age'] - 55) / 15 * 0.3 +
    (final_df['systolic_bp'] - 125) / 15 * 0.25 +
    (final_df['glucose_level'] - 100) / 20 * 0.2 +
    (final_df['bmi'] - 28) / 5 * 0.15 +
    final_df['smoker_status'].map({'yes': 0.1, 'former': 0.05, 'no': 0})
)

# Convert risk score to probability using sigmoid function
# Adjust intercept (-2.5 here) to control the overall prevalence (imbalance)
prob = 1 / (1 + np.exp(-(risk_score - 2.5)))

# Generate outcome based on probability
final_df['disease_outcome'] = (np.random.rand(len(final_df)) < prob).astype(int)

# --- Final Checks and Save ---
print(f"\nGenerated {len(final_df)} records for {N_PATIENTS} patients.")
print("\nTarget variable distribution ('disease_outcome'):")
print(final_df['disease_outcome'].value_counts(normalize=True))

# Select and order final columns
final_columns = [
    'patient_id', 'timestamp', 'age', 'systolic_bp', 'diastolic_bp',
    'glucose_level', 'bmi', 'smoker_status', 'heart_rate', 'disease_outcome'
]
final_df = final_df[final_columns]

# Save to CSV
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSynthetic data saved to {OUTPUT_FILE}")