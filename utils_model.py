import numpy as np
from sklearn.cluster import KMeans
import joblib

class CreditModel:
    def __init__(self, model_path="credit_kmeans_model.pkl"):
        """
        Initialize the CreditModel class.
        
        :param model_path: Path to the saved KMeans model file.
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained KMeans model from the specified file.
        """
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise Exception("Model file not found. Train the model first using 'train_model.py'.")
    
    def predict(self, X):
        """
        Predict credit category tiers using the loaded model.
        
        :param X: Feature data as a numpy array or pandas DataFrame.
        :return: Array of predicted cluster labels.
        """
        if self.model is None:
            raise Exception("Model is not loaded.")
        return self.model.predict(X)
    
    def fine_tune(self, X, n_clusters=4, n_init=10):
        """
        Fine-tune the KMeans model with new data.
        
        :param X: Feature data as a numpy array or pandas DataFrame.
        :param n_clusters: Number of clusters for KMeans.
        :param n_init: Number of initializations for KMeans.
        """
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        self.model.fit(X)
        joblib.dump(self.model, self.model_path)
        print("Model fine-tuned and saved successfully.")

# Instructions to predict credit category tiers
if __name__ == "__main__":
    # Example of usage
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Define the path to the dataset and features
    dataset_path = "credit_scoring.csv"
    features = ['Credit Utilization Ratio', 'Payment History', 
                'Number of Credit Accounts', 'Loan Amount', 'Interest Rate']
    
    # Load and preprocess the data
    df = pd.read_csv(dataset_path)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    # Initialize the model and predict
    model = CreditModel(model_path="credit_kmeans_model.pkl")
    predictions = model.predict(X)
    
    # Add predictions to the dataset and print the result
    df['Credit Category Tier'] = predictions
    print(df[['Credit Category Tier']])
