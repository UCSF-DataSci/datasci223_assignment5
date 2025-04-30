```python {"tags": ["setup"]}
# Install necessary packages
# Ensure you have a requirements.txt file in the root directory
%pip install -r requirements.txt
```
# Part 2: Time Series Features & Tree-Based Models

**Objective:** Extract basic rolling window features from the time-series data (`heart_rate`), train Random Forest and XGBoost models using these features, and compare their performance.

## 1. Setup

Import necessary libraries. We'll need libraries from Part 1, plus RandomForest, XGBoost, and potentially others for feature engineering.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb # Make sure xgboost is installed (pip install xgboost)
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# Assuming load_data and calculate_evaluation_metrics might be useful
# If they are in part1_introduction.ipynb, you might need to copy them here
# or import them if you structure this as a module (more advanced).
# For simplicity in a homework setting, let's assume we might redefine or copy load_data if needed.

# --- Copy or Import load_data if needed ---
# Example: Redefining load_data for self-containment
def load_data(file_path: str) -> pd.DataFrame:
    """Loads the synthetic health data from a CSV file."""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp']) # Ensure timestamp is parsed
        if df.empty:
            print(f"Warning: Loaded empty DataFrame from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()
# ------------------------------------------

# Define the file path
DATA_FILE = 'data/synthetic_health_data.csv'
df_part2 = load_data(DATA_FILE)

# Display basic info
if not df_part2.empty:
    print(df_part2.info())
    print(df_part2.head())
```

## 2. Feature Engineering: Rolling Features

Implement `extract_rolling_features` to calculate rolling mean and standard deviation for the `heart_rate`.

<!-- #region {"tags": ["task"]} -->
```python
def extract_rolling_features(df: pd.DataFrame, window_size_seconds: int) -> pd.DataFrame:
    """Calculates rolling mean and std dev for heart_rate based on time window.

    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp' and 'heart_rate'.
        window_size_seconds (int): The window size in seconds.

    Returns:
        pd.DataFrame: DataFrame with added 'hr_rolling_mean' and 'hr_rolling_std'.
                      Returns an empty DataFrame if required columns are missing.
    """
    # --- YOUR CODE HERE ---
    if 'timestamp' not in df.columns or 'heart_rate' not in df.columns:
        print("Error: DataFrame must contain 'timestamp' and 'heart_rate' columns.")
        return pd.DataFrame()

    # Ensure data is sorted by timestamp for correct rolling calculation
    df_sorted = df.sort_values('timestamp').copy()

    # Set timestamp as index for time-based rolling
    df_sorted.set_index('timestamp', inplace=True)

    # Define the window size as a timedelta string
    window = f'{window_size_seconds}s'

    # Calculate rolling features
    # min_periods=1 ensures calculation starts even if window isn't full
    df_sorted['hr_rolling_mean'] = df_sorted['heart_rate'].rolling(window, min_periods=1).mean()
    df_sorted['hr_rolling_std'] = df_sorted['heart_rate'].rolling(window, min_periods=1).std()

    # Reset index to bring timestamp back as a column
    df_sorted.reset_index(inplace=True)

    # Fill initial NaNs in std dev (first value has no std dev)
    df_sorted['hr_rolling_std'].fillna(0, inplace=True)

    return df_sorted
    # ----------------------
```
<!-- #endregion -->

```python
if not df_part2.empty:
    # Example: Use a 5-minute window (300 seconds)
    WINDOW_SIZE = 300
    df_featured = extract_rolling_features(df_part2, WINDOW_SIZE)

    print(f"\nDataFrame with rolling features (window={WINDOW_SIZE}s):")
    print(df_featured[['timestamp', 'heart_rate', 'hr_rolling_mean', 'hr_rolling_std']].head())
    print(df_featured[['timestamp', 'heart_rate', 'hr_rolling_mean', 'hr_rolling_std']].tail())
    print("\nCheck for NaNs introduced:")
    print(df_featured[['hr_rolling_mean', 'hr_rolling_std']].isnull().sum())

```

## 3. Data Preparation for Tree Models

Implement `prepare_data_part2` using the newly engineered features.

<!-- #region {"tags": ["task"]} -->
```python
def prepare_data_part2(df_with_features: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepares data for Part 2: selects features (incl. rolling), splits, imputes.

    Args:
        df_with_features (pd.DataFrame): DataFrame with original and rolling features.
        test_size (float): Proportion for the test set.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # --- YOUR CODE HERE ---
    # Select features: original relevant features + new rolling features
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi',
                'hr_rolling_mean', 'hr_rolling_std'] # Add rolling features
    target = 'disease_outcome'

    # Drop rows where rolling features might be NaN (if any remained after fillna(0) for std)
    # This typically happens at the very beginning of the series if min_periods > 1 was used
    df_clean = df_with_features.dropna(subset=features).copy()

    if not all(f in df_clean.columns for f in features) or target not in df_clean.columns:
         raise ValueError("Required columns are missing from the DataFrame after cleaning NaNs.")

    X = df_clean[features]
    y = df_clean[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Handle potential missing values in original features (if any)
    # Using median imputation again for consistency
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrames
    X_train = pd.DataFrame(X_train_imputed, columns=features, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=features, index=X_test.index)

    return X_train, X_test, y_train, y_test
    # ----------------------
```
<!-- #endregion -->

```python
if 'df_featured' in locals() and not df_featured.empty:
    X_train_pt2, X_test_pt2, y_train_pt2, y_test_pt2 = prepare_data_part2(df_featured)
    print("\nData prepared for Part 2:")
    print("X_train_pt2 shape:", X_train_pt2.shape)
    print("X_test_pt2 shape:", X_test_pt2.shape)
    print("Features:", X_train_pt2.columns.tolist())
```

## 4. Model Training: Random Forest

Implement `train_random_forest`.

<!-- #region {"tags": ["task"]} -->
```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Trains a RandomForestClassifier model.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.
        random_state (int): Random seed.

    Returns:
        RandomForestClassifier: The trained model.
    """
    # --- YOUR CODE HERE ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1 # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    return model
    # ----------------------
```
<!-- #endregion -->

```python
if 'X_train_pt2' in locals():
    rf_model = train_random_forest(X_train_pt2, y_train_pt2)
    print("\nRandom Forest model trained:", rf_model)
```

## 5. Model Training: XGBoost

Implement `train_xgboost`.

<!-- #region {"tags": ["task"]} -->
```python
def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """Trains an XGBClassifier model.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage.
        max_depth (int): Maximum depth of a tree.
        random_state (int): Random seed.

    Returns:
        xgb.XGBClassifier: The trained model.
    """
    # --- YOUR CODE HERE ---
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False, # Recommended to avoid warnings
        eval_metric='logloss' # Common metric for binary classification
    )
    model.fit(X_train, y_train)
    return model
    # ----------------------
```
<!-- #endregion -->

```python
if 'X_train_pt2' in locals():
    xgb_model = train_xgboost(X_train_pt2, y_train_pt2)
    print("\nXGBoost model trained:", xgb_model)
```

## 6. Model Comparison

Calculate the ROC AUC score for both models on the test set and compare them.

```python
if 'rf_model' in locals() and 'xgb_model' in locals():
    # Predict probabilities for AUC calculation
    rf_probs = rf_model.predict_proba(X_test_pt2)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test_pt2)[:, 1]

    # Calculate AUC
    rf_auc = roc_auc_score(y_test_pt2, rf_probs)
    xgb_auc = roc_auc_score(y_test_pt2, xgb_probs)

    print(f"\nModel Comparison (AUC on Test Set):")
    print(f"  Random Forest: {rf_auc:.4f}")
    print(f"  XGBoost:       {xgb_auc:.4f}")

    # Optional: Feature Importances (can be insightful)
    if hasattr(rf_model, 'feature_importances_'):
        rf_importances = pd.Series(rf_model.feature_importances_, index=X_train_pt2.columns).sort_values(ascending=False)
        print("\nRandom Forest Feature Importances:")
        print(rf_importances)

    if hasattr(xgb_model, 'feature_importances_'):
        xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train_pt2.columns).sort_values(ascending=False)
        print("\nXGBoost Feature Importances:")
        print(xgb_importances)

```

## 7. Interpretation

*(Markdown Cell for Students)*

**Compare the performance of the Random Forest and XGBoost models based on their AUC scores.**

*   Which model performed better on this task with the added time-series features?
*   Look at the feature importances (if calculated). Do the rolling heart rate features seem important for either model?
*   How do these AUC scores compare to the Logistic Regression model's AUC from Part 1 (you might need to run Part 1 again or note the value)? Did adding the rolling features improve performance significantly?
## 8. Save Results

Save the calculated AUC scores to a text file.

```python
import os

# Define output directory and file
RESULTS_DIR = 'results'
output_file_part2 = os.path.join(RESULTS_DIR, 'results_part2.txt')

# Create directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if AUC scores exist
if 'rf_auc' in locals() and 'xgb_auc' in locals():
    # Format results
    lines_to_write = [
        f"rf_auc: {rf_auc:.4f}",
        f"xgb_auc: {xgb_auc:.4f}"
    ]

    # Write to file
    try:
        with open(output_file_part2, 'w') as f:
            f.write("\n".join(lines_to_write))
        print(f"Part 2 results saved to {output_file_part2}")
    except Exception as e:
        print(f"Error saving Part 2 results: {e}")
else:
    print("AUC results (rf_auc, xgb_auc) not found. Skipping saving Part 2 results.")

```