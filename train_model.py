import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def preprocess_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Feature engineering: derive a 'Credit Score' or scale relevant features
    scaler = StandardScaler()
    features = ['Credit Utilization Ratio', 'Payment History', 
                'Number of Credit Accounts', 'Loan Amount', 'Interest Rate']
    df_scaled = scaler.fit_transform(df[features])
    
    return df_scaled

def train_model(data_path, model_path="credit_kmeans_model.pkl"):
    # Preprocess the data
    X = preprocess_data(data_path)
    
    # Train the KMeans model
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    kmeans.fit(X)
    
    # Save the model
    joblib.dump(kmeans, model_path)
    print(f"Model trained and saved to {model_path}.")
    
if __name__ == "__main__":
    # Replace 'credit_scoring.csv' with the dataset path
    train_model("credit_scoring.csv")
