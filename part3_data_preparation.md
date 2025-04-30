# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 3: Practical Data Preparation

**Objective:** Handle categorical features using One-Hot Encoding and address class imbalance using SMOTE.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
```

## 2. Data Loading

Load the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    # YOUR CODE HERE
    # Load the CSV file using pandas
    
    return pd.DataFrame()  # Replace with actual implementation
```

## 3. Categorical Feature Encoding

Implement `encode_categorical_features` using `OneHotEncoder`.

```python
def encode_categorical_features(df, column_to_encode='smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.
    
    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode
        
    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    # YOUR CODE HERE
    # 1. Extract the categorical column
    # 2. Apply OneHotEncoder
    # 3. Create new column names
    # 4. Replace the original categorical column with the encoded columns
    
    # Placeholder return - replace with your implementation
    return df.copy()
```

## 4. Data Preparation

Implement `prepare_data_part3` to handle the train/test split correctly.

```python
def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    # 1. Encode categorical features using the encode_categorical_features function
    # 2. Select relevant features (including the one-hot encoded ones) and the target
    # 3. Split data into training and testing sets
    # 4. Handle missing values
    
    # Placeholder return - replace with your implementation
    return None, None, None, None
```

## 5. Handling Imbalanced Data

Implement `apply_smote` to oversample the minority class.

```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    # YOUR CODE HERE
    # Apply SMOTE to balance the classes
    
    # Placeholder return - replace with your implementation
    return X_train, y_train
```

## 6. Model Training and Evaluation

Train a model on the SMOTE-resampled data and evaluate it.

```python
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # YOUR CODE HERE
    # Initialize and train a LogisticRegression model
    
    return None  # Replace with actual implementation

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # YOUR CODE HERE
    # 1. Generate predictions
    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    # 3. Create confusion matrix
    # 4. Return metrics in a dictionary
    
    # Placeholder return - replace with your implementation
    return {}
```

## 7. Save Results

Save the evaluation metrics to a text file.

```python
# YOUR CODE HERE
# 1. Create 'results' directory if it doesn't exist
# 2. Format metrics as strings
# 3. Write metrics to 'results/results_part3.txt'
```

## 8. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 7. Save results
    # (Your code for saving results)
    
    # 8. Load Part 1 results for comparison
    import json
    try:
        with open('results/results_part1.txt', 'r') as f:
            part1_metrics = json.load(f)
        
        # 9. Compare models
        comparison = compare_models(part1_metrics, metrics)
        print("\nModel Comparison (improvement percentages):")
        for metric, improvement in comparison.items():
            print(f"{metric}: {improvement:.2f}%")
    except FileNotFoundError:
        print("Part 1 results not found. Run part1_introduction.ipynb first.")
```

## 9. Compare Results

Implement a function to compare model performance between balanced and imbalanced data.

```python
def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    # YOUR CODE HERE
    # 1. Calculate percentage improvement for each metric
    # 2. Handle metrics where higher is better (most metrics) and where lower is better
    # 3. Return a dictionary with metric names and improvement percentages
    
    # Placeholder return - replace with your implementation
    return {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auc': 0.0
    }