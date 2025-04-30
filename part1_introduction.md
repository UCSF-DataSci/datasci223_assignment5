```python {"tags": ["setup"]}
# Install necessary packages
# Ensure you have a requirements.txt file in the root directory
%pip install -r requirements.txt
```
# Part 1: Introduction to Classification & Evaluation

**Objective:** Load the synthetic health data, perform basic preparation, train a Logistic Regression model, and evaluate its performance using standard classification metrics.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Configure plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## 2. Data Loading

Implement the `load_data` function to read the dataset.

<!-- #region {"tags": ["task"]} -->
```python
def load_data(file_path: str) -> pd.DataFrame:
    """Loads the synthetic health data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    # --- YOUR CODE HERE ---
    try:
        df = pd.read_csv(file_path)
        # Optional: Basic validation (e.g., check if empty)
        if df.empty:
            print(f"Warning: Loaded empty DataFrame from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        # Return an empty DataFrame or raise an error, depending on desired handling
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()
    # ----------------------
```
<!-- #endregion -->

```python
# Define the file path (assuming it's in a 'data' directory)
# You might need to run the data generation script first if it doesn't exist
DATA_FILE = 'data/synthetic_health_data.csv'

df = load_data(DATA_FILE)

# Display basic info and first few rows
if not df.empty:
    print(df.info())
    print(df.head())
```

## 3. Data Preparation

Implement `prepare_data_part1` to select features, split data, and handle missing values.

<!-- #region {"tags": ["task"]} -->
```python
def prepare_data_part1(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepares data for Part 1: selects features, splits, imputes missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): Proportion of data to use for the test set.
        random_state (int): Random seed for splitting.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # --- YOUR CODE HERE ---
    # Select features and target
    # Note: Exclude 'smoker_status' (categorical) and 'heart_rate' (used in Part 2) for this part
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    target = 'disease_outcome'

    if not all(f in df.columns for f in features) or target not in df.columns:
         raise ValueError("Required columns are missing from the DataFrame")

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify for imbalanced data
    )

    # Handle missing values (using median imputation)
    # Fit imputer ONLY on training data
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    # Transform test data using the SAME imputer
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrames (important for column names)
    X_train = pd.DataFrame(X_train_imputed, columns=features, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=features, index=X_test.index)

    return X_train, X_test, y_train, y_test
    # ----------------------
```
<!-- #endregion -->

```python
if not df.empty:
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    print("Data prepared:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:\n", y_train.value_counts(normalize=True))
    print("y_test distribution:\n", y_test.value_counts(normalize=True))
```

## 4. Model Training: Logistic Regression

Implement `train_logistic_regression`.

<!-- #region {"tags": ["task"]} -->
```python
def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        LogisticRegression: The trained model.
    """
    # --- YOUR CODE HERE ---
    model = LogisticRegression(random_state=42, max_iter=1000) # Increase max_iter if needed
    model.fit(X_train, y_train)
    return model
    # ----------------------
```
<!-- #endregion -->

```python
if 'X_train' in locals(): # Check if data preparation was successful
    log_reg_model = train_logistic_regression(X_train, y_train)
    print("Logistic Regression model trained:", log_reg_model)
```

## 5. Model Evaluation

Implement `calculate_evaluation_metrics` to assess the model's performance.

<!-- #region {"tags": ["task"]} -->
```python
def calculate_evaluation_metrics(model, X_test, y_test):
    """Calculates various classification metrics.

    Args:
        model: The trained classification model.
        X_test: Test features.
        y_test: Test target.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, f1, auc,
              and confusion_matrix.
    """
    # --- YOUR CODE HERE ---
    y_pred = model.predict(X_test)
    # Use try-except for predict_proba as not all models have it (though LogisticRegression does)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except AttributeError:
        print("Model does not have predict_proba method. AUC cannot be calculated.")
        auc_score = np.nan # Or handle as appropriate

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': auc_score,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist() # Convert to list for easier handling/testing
    }
    return metrics
    # ----------------------
```
<!-- #endregion -->

```python
if 'log_reg_model' in locals(): # Check if model training was successful
    evaluation_results = calculate_evaluation_metrics(log_reg_model, X_test, y_test)
    print("\nEvaluation Metrics:")
    for metric, value in evaluation_results.items():
        if metric == 'confusion_matrix':
            print(f"  {metric}:")
            # Pretty print confusion matrix
            cm = np.array(value)
            print(f"    TN: {cm[0,0]}  FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}  TP: {cm[1,1]}")
        elif isinstance(value, float):
             print(f"  {metric}: {value:.4f}")
        else:
             print(f"  {metric}: {value}")

```

## 6. Interpretation

*(Markdown Cell for Students)*

**Discuss the evaluation metrics calculated above.**

*   Which metric seems most informative given the potential for class imbalance in health data (where correctly identifying the 'disease' class is often crucial)?
*   What does the confusion matrix tell you about the types of errors the model is making (False Positives vs. False Negatives)?
*   How does the AUC score compare to a random guess (AUC=0.5)?

*(Remember, we haven't explicitly addressed the imbalance yet - that comes in Part 3. This is just an initial assessment.)*
## 7. Save Results

Save the calculated evaluation metrics to a text file.

```python
import os

# Define output directory and file
RESULTS_DIR = 'results'
output_file_part1 = os.path.join(RESULTS_DIR, 'results_part1.txt')

# Create directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if results exist
if 'evaluation_results' in locals():
    # Format results (excluding confusion matrix)
    lines_to_write = []
    for metric, value in evaluation_results.items():
        if metric != 'confusion_matrix':
            lines_to_write.append(f"{metric}: {value:.4f}") # Format floats

    # Write to file
    try:
        with open(output_file_part1, 'w') as f:
            f.write("\n".join(lines_to_write))
        print(f"Part 1 results saved to {output_file_part1}")
    except Exception as e:
        print(f"Error saving Part 1 results: {e}")
else:
    print("Evaluation results not found. Skipping saving Part 1 results.")

```