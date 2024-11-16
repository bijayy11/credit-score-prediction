import pandas as pd
import numpy as np

# Load the original dataset to understand patterns for autofilling
def load_original_dataset(data_path="credit_scoring.csv"):
    """
    Load the original dataset to analyze patterns and distributions.
    """
    return pd.read_csv(data_path)

# Generate a new row based on entered values
def generate_dataset(user_inputs, data_path="credit_scoring.csv"):
    """
    Create a dataset with user-specified fields and autofill the rest.

    :param user_inputs: Dictionary with user-provided field values.
    :param data_path: Path to the original dataset.
    :return: A pandas DataFrame with the generated row.
    """
    # Load the original dataset
    df = load_original_dataset(data_path)
    
    # Autofill logic: Fill missing fields with column means (for numerical) or most frequent values (for categorical)
    new_row = {}
    for column in df.columns:
        if column in user_inputs:
            new_row[column] = user_inputs[column]
        elif df[column].dtype in [np.float64, np.int64]:  # Numerical columns
            new_row[column] = df[column].mean()
        else:  # Categorical columns
            new_row[column] = df[column].mode()[0]
    
    return pd.DataFrame([new_row])

# Example of how to use the generate_dataset function
if __name__ == "__main__":
    # User-provided inputs
    user_inputs = {
        "Age": 35,
        "Gender": "Male",
        "Marital Status": "Married",
        "Credit Utilization Ratio": 0.5
    }
    
    # Path to the original dataset
    dataset_path = "credit_scoring.csv"
    
    # Generate the new dataset
    generated_data = generate_dataset(user_inputs, dataset_path)
    
    # Display the result
    print("Generated Dataset:")
    print(generated_data)
