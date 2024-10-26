import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Database connection details
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'motor_db'
DB_USER = 'motor_user'
DB_PASSWORD = 'Admin'

# Create an SQLAlchemy engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# SQL query to fetch historical data for testing
QUERY_HISTORICAL_DATA = """
    SELECT temperature, vibration, current, failure
    FROM motor_sensor_data
    WHERE timestamp < NOW() - INTERVAL '10 minutes';
"""

def fetch_historical_data():
    """
    Fetches historical sensor data from the database.
    
    Returns:
    pd.DataFrame: A DataFrame containing historical sensor data.
    """
    return pd.read_sql(QUERY_HISTORICAL_DATA, engine)

def train_and_evaluate_model(data):
    """
    Trains and evaluates a RandomForest model on historical data.
    
    Args:
    data (pd.DataFrame): The dataset containing features and target.

    Returns:
    tuple: Accuracy, confusion matrix, classification report, predictions, true labels
    """
    # Split data into features (X) and target (y)
    X = data[['temperature', 'vibration', 'current']]
    y = data['failure']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with scaling and RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, conf_matrix, class_report, y_pred, y_test

def plot_confusion_matrix(conf_matrix):
    """
    Plots the confusion matrix.
    
    Args:
    conf_matrix (np.ndarray): Confusion matrix array.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'], 
                yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

def plot_classification_report(class_report):
    """
    Plots the classification report as a heatmap.
    
    Args:
    class_report (dict): Classification report dictionary.
    """
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(class_report).transpose()

    # Plot the classification report
    plt.figure(figsize=(10, 7))
    sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f',
                xticklabels=['Precision', 'Recall', 'F1-Score', 'Support'], 
                yticklabels=report_df.index[:-1])
    plt.title('Classification Report')
    plt.savefig('classification_report.png')

def main():
    """
    Main function to load data, train the model, evaluate performance, and visualize results.
    """
    # Fetch historical data
    data = fetch_historical_data()
    
    if data.empty:
        print("No historical data found. Please ensure the data is available.")
        return

    # Train and evaluate the model
    accuracy, conf_matrix, class_report, y_pred, y_test = train_and_evaluate_model(data)
    
    # Print performance metrics
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)
    
    # Plot classification report
    plot_classification_report(class_report)

if __name__ == "__main__":
    main()
