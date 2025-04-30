```python {"tags": ["setup"]}
# Install necessary packages
# Ensure you have a requirements.txt file in the root directory
%pip install -r requirements.txt
```
# Part 3: Practical Data Preparation

**Objective:** Handle categorical features using One-Hot Encoding and address class imbalance in the target variable using SMOTE. Evaluate the impact of these techniques on the Logistic Regression model's performance.

## 1. Setup

Import necessary libraries, including `OneHotEncoder` and `SMOTE`.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE # Make sure imbalanced-learn is installed (pip install imbalanced-learn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # Ignore potential future warnings from sklearn/imblearn

# --- Copy or Import load_data and calculate_evaluation_metrics if needed ---
# Example: Redefining load_data
def load_data(file_path: str) -> pd.DataFrame:
    """Loads the synthetic health data from a CSV file."""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        if df.empty:
            print(f"Warning: Loaded empty DataFrame from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

# Example: Redefining calculate_evaluation_metrics
def calculate_evaluation_metrics(model, X_test, y_test):
    """Calculates various classification metrics."""
    y_pred = model.predict(X_test)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except AttributeError:
        auc_score = np.nan

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': auc_score,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

# Example: Redefining train_logistic_regression
def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model
# --------------------------------------------------------------------

# Define the file path
DATA_FILE = 'data/synthetic_health_data.csv'
df_part3 = load_data(DATA_FILE)

# Display basic info
if not df_part3.empty:
    print(df_part3.info())
    print("Original 'smoker_status' values:")
    print(df_part3['smoker_status'].value_counts())
```

## 2. Categorical Feature Encoding

Implement `encode_categorical_features` using `OneHotEncoder`. Remember the importance of fitting *only* on training data later in the pipeline, but for this function, we'll demonstrate the encoding process. The actual fitting/transforming separation happens within `prepare_data_part3`.

<!-- #region {"tags": ["task"]} -->
```python
def encode_categorical_features(df: pd.DataFrame, column_to_encode: str = 'smoker_status') -> pd.DataFrame:
    """Encodes a specified categorical column using OneHotEncoder.

    Note: This function demonstrates encoding. In a real pipeline, fit the encoder
    on the training set and transform both train and test sets separately.
    This function applies fit_transform directly for simplicity here, but
    the proper split handling is done in prepare_data_part3.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_to_encode (str): Name of the categorical column to encode.

    Returns:
        pd.DataFrame: DataFrame with the original categorical column replaced
                      by one-hot encoded columns. Returns original df if column missing.
    """
    # --- YOUR CODE HERE ---
    if column_to_encode not in df.columns:
        print(f"Warning: Column '{column_to_encode}' not found. Returning original DataFrame.")
        return df

    df_encoded = df.copy()
    categorical_column = df_encoded[[column_to_encode]]

    # Initialize OneHotEncoder
    # handle_unknown='ignore' prevents errors if test set has categories not seen in train
    # sparse=False returns a dense numpy array, easier to work with
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and transform the categorical column
    # In a real pipeline, fit ONLY on train data
    encoded_data = encoder.fit_transform(categorical_column)

    # Create new column names for the encoded features
    # Example: smoker_status_no, smoker_status_yes, smoker_status_former
    encoded_columns = encoder.get_feature_names_out([column_to_encode])

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_encoded.index)

    # Drop the original categorical column
    df_encoded = df_encoded.drop(column_to_encode, axis=1)

    # Concatenate the original DataFrame (without the categorical column) and the encoded DataFrame
    df_final = pd.concat([df_encoded, encoded_df], axis=1)

    return df_final
    # ----------------------
```
<!-- #endregion -->

```python
if not df_part3.empty:
    df_encoded_example = encode_categorical_features(df_part3)
    print("\nDataFrame after OneHotEncoding 'smoker_status':")
    print(df_encoded_example.info())
    print(df_encoded_example.head())
    # Display names of new columns
    print("\nNew encoded columns:")
    print([col for col in df_encoded_example.columns if 'smoker_status_' in col])
```

## 3. Data Preparation with Encoding

Implement `prepare_data_part3`. This function *must* handle the train/test split *before* fitting the encoder to prevent data leakage.

<!-- #region {"tags": ["task"]} -->
```python
def prepare_data_part3(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepares data for Part 3: selects features, encodes categoricals (correctly), splits, imputes.

    Args:
        df (pd.DataFrame): The input DataFrame (original, before encoding).
        test_size (float): Proportion for the test set.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test, encoder (the fitted OneHotEncoder)
    """
    # --- YOUR CODE HERE ---
    # Select features (including the categorical one) and target
    # Exclude heart_rate features from Part 2 for this part
    features_numeric = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    feature_categorical = 'smoker_status'
    target = 'disease_outcome'

    all_features = features_numeric + [feature_categorical]

    if not all(f in df.columns for f in all_features) or target not in df.columns:
         raise ValueError("Required columns are missing from the DataFrame")

    X = df[all_features]
    y = df[target]

    # 1. Split data FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Handle missing values in NUMERIC features (before encoding)
    imputer = SimpleImputer(strategy='median')
    X_train[features_numeric] = imputer.fit_transform(X_train[features_numeric])
    X_test[features_numeric] = imputer.transform(X_test[features_numeric])
    # Note: Imputation for categorical might need 'most_frequent' strategy if needed,
    # but OneHotEncoder can handle NaNs if configured, or they can be dropped.
    # Assuming 'smoker_status' has no NaNs for simplicity, or handle as needed.
    X_train.dropna(subset=[feature_categorical], inplace=True)
    y_train = y_train[X_train.index] # Align target
    X_test.dropna(subset=[feature_categorical], inplace=True)
    y_test = y_test[X_test.index] # Align target


    # 3. Fit OneHotEncoder ONLY on Training data
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[[feature_categorical]])

    # 4. Transform both Training and Test data
    encoded_train = encoder.transform(X_train[[feature_categorical]])
    encoded_test = encoder.transform(X_test[[feature_categorical]])

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out([feature_categorical])

    # Create DataFrames for encoded features
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_columns, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_columns, index=X_test.index)

    # Combine numeric and encoded features
    X_train_final = pd.concat([X_train[features_numeric], encoded_train_df], axis=1)
    X_test_final = pd.concat([X_test[features_numeric], encoded_test_df], axis=1)

    return X_train_final, X_test_final, y_train, y_test, encoder
    # ----------------------
```
<!-- #endregion -->

```python
if not df_part3.empty:
    X_train_pt3, X_test_pt3, y_train_pt3, y_test_pt3, fitted_encoder = prepare_data_part3(df_part3)
    print("\nData prepared for Part 3 (with OneHotEncoding):")
    print("X_train_pt3 shape:", X_train_pt3.shape)
    print("X_test_pt3 shape:", X_test_pt3.shape)
    print("Features:", X_train_pt3.columns.tolist())
    print("y_train_pt3 distribution:\n", y_train_pt3.value_counts(normalize=True))
```

## 4. Handling Imbalanced Data with SMOTE

Implement `apply_smote`. This should *only* be applied to the training data.

<!-- #region {"tags": ["task"]} -->
```python
def apply_smote(X_train, y_train, random_state=42):
    """Applies SMOTE to oversample the minority class in the training data.

    Args:
        X_train: Training features.
        y_train: Training target.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train_resampled, y_train_resampled
    """
    # --- YOUR CODE HERE ---
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled
    # ----------------------
```
<!-- #endregion -->

```python
if 'X_train_pt3' in locals():
    print("\nOriginal training distribution:")
    print(y_train_pt3.value_counts(normalize=True))

    X_train_res, y_train_res = apply_smote(X_train_pt3, y_train_pt3)

    print("\nResampled training distribution (after SMOTE):")
    print("X_train_res shape:", X_train_res.shape)
    print(y_train_res.value_counts(normalize=True))
```

## 5. Retrain and Evaluate Model

Retrain the Logistic Regression model using the SMOTE-resampled training data and evaluate it on the original, unbalanced test set.

```python
if 'X_train_res' in locals():
    # Retrain Logistic Regression on balanced data
    log_reg_model_smote = train_logistic_regression(X_train_res, y_train_res)
    print("\nLogistic Regression model trained on SMOTE data:", log_reg_model_smote)

    # Evaluate on the ORIGINAL UNBALANCED test set
    evaluation_results_smote = calculate_evaluation_metrics(log_reg_model_smote, X_test_pt3, y_test_pt3)

    print("\nEvaluation Metrics (Model trained on SMOTE data, evaluated on original test set):")
    for metric, value in evaluation_results_smote.items():
        if metric == 'confusion_matrix':
            print(f"  {metric}:")
            cm = np.array(value)
            print(f"    TN: {cm[0,0]}  FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}  TP: {cm[1,1]}")
        elif isinstance(value, float):
             print(f"  {metric}: {value:.4f}")
        else:
             print(f"  {metric}: {value}")

    # Optional: Compare with Part 1 results if available
    # print("\nCompare with Part 1 Metrics (if available):")
    # print(evaluation_results) # Assuming 'evaluation_results' holds Part 1 metrics
```

## 6. Interpretation

*(Markdown Cell for Students)*

**Compare the evaluation metrics of the model trained on the SMOTE-balanced data (evaluated on the original test set) with the metrics from the model in Part 1 (trained and evaluated on the original imbalanced data).**

*   How did SMOTE affect the different metrics, particularly `recall`, `precision`, and `f1-score`?
*   Did balancing the training data improve the model's ability to identify the minority class (likely the 'disease' class)? Explain why or why not, referring to the confusion matrix changes.
*   Considering the goal of identifying disease cases, was applying SMOTE beneficial in this context?
## 7. Save Results

Save the evaluation metrics from the SMOTE-trained model to a text file.

```python
import os

# Define output directory and file
RESULTS_DIR = 'results'
output_file_part3 = os.path.join(RESULTS_DIR, 'results_part3.txt')

# Create directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if results exist
if 'evaluation_results_smote' in locals():
    # Format results (excluding confusion matrix)
    lines_to_write = []
    for metric, value in evaluation_results_smote.items():
        if metric != 'confusion_matrix':
             # Handle potential NaN for AUC if predict_proba wasn't available
             if isinstance(value, float) and not np.isnan(value):
                 lines_to_write.append(f"{metric}: {value:.4f}")
             else:
                 lines_to_write.append(f"{metric}: {value}") # Keep non-float as is (or NaN)


    # Write to file
    try:
        with open(output_file_part3, 'w') as f:
            f.write("\n".join(lines_to_write))
        print(f"Part 3 results saved to {output_file_part3}")
    except Exception as e:
        print(f"Error saving Part 3 results: {e}")
else:
    print("SMOTE evaluation results not found. Skipping saving Part 3 results.")

```