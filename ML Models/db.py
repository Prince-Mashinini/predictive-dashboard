from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection details
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'motor_db'
DB_USER = 'motor_user'
DB_PASSWORD = 'Admin'

# Create an SQLAlchemy engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Function to fetch historical data for training
def fetch_historical_data():
    query = """
        SELECT temperature, vibration, current, failure
        FROM motor_sensor_data
        WHERE timestamp < NOW() - INTERVAL '1 minutes';
    """
    return pd.read_sql(query, engine)

# Function to fetch the latest prediction timestamp
def fetch_latest_prediction_timestamp():
    query = """
        SELECT MAX(timestamp) AS latest_timestamp
        FROM motor_failure_predictions
    """
    result = pd.read_sql(query, engine)
    latest_timestamp = result['latest_timestamp'].iloc[0]
    return latest_timestamp

# Function to fetch new sensor data after the latest prediction timestamp
def fetch_sensor_data(latest_prediction_timestamp):
    # Fetch only new sensor data that has not been predicted yet
    query = f"""
        SELECT timestamp, motor_id, temperature, vibration, current
        FROM motor_sensor_data
        WHERE timestamp > '{latest_prediction_timestamp}'
        ORDER BY timestamp ASC;
    """
    return pd.read_sql(query, engine)

# Function to suggest maintenance tasks based on prediction
def suggest_maintenance_tasks(failure_probability):
    if failure_probability > 0.8:
        return "Immediate action required: Inspect motor, replace bearings, lubricate, and run full diagnostics."
    elif 0.5 < failure_probability <= 0.8:
        return "High priority: Perform detailed inspection, check for overheating, and lubricate motor."
    elif 0.2 < failure_probability <= 0.5:
        return "Medium priority: Inspect for unusual vibrations or noise, check alignment, and perform routine maintenance."
    else:
        return "Low priority: Routine check-up and basic maintenance."

# SQL query to insert predictions and maintenance tasks into the database
INSERT_PREDICTION_QUERY = """
    INSERT INTO motor_failure_predictions (timestamp, motor_id, failure_probability, prediction, maintenance_tasks, acknowledged)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

# Function to store predictions and maintenance tasks in the database
def store_predictions(engine, insert_data):
    """
    Stores the prediction results and suggested maintenance tasks into the database.
    
    Args:
    engine: SQLAlchemy engine instance.
    insert_data (list): List of tuples containing prediction data to be inserted.
    """
    insert_query = text("""
        INSERT INTO motor_failure_predictions (timestamp, motor_id, failure_probability, prediction, maintenance_tasks, acknowledged)
        VALUES (:timestamp, :motor_id, :failure_probability, :prediction, :maintenance_tasks, :acknowledged)
    """)

    with engine.connect() as conn:
        with conn.begin():
            try:
                conn.execute(insert_query, [
                    {
                        'timestamp': data[0],
                        'motor_id': int(data[1]),
                        'failure_probability': float(data[2]),
                        'prediction': bool(data[3]),
                        'maintenance_tasks': data[4],
                        'acknowledged': False  # Default to False until user acknowledges it
                    }
                    for data in insert_data
                ])
                logging.info(f"Inserted {len(insert_data)} predictions and maintenance tasks into the database.")
            except Exception as e:
                logging.error(f"Failed to insert predictions: {e}")


# Train a RandomForest model on historical data
def train_model():
    data = fetch_historical_data()

    if data.empty:
        logging.warning("No data found in the historical dataset. Please check the data availability.")
        return None  # Return None instead of raising an error

    # Proceed with training if data is available
    X = data[['temperature', 'vibration', 'current']]
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Model training completed. Accuracy: {accuracy * 100:.2f}%')

    return pipeline

# Main function to perform predictions in real-time
def main():
    # Train the model on historical data (once)
    model = train_model()

    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    try:
        while True:
            # Fetch the latest prediction timestamp from the database
            latest_prediction_timestamp = fetch_latest_prediction_timestamp()
            
            if latest_prediction_timestamp is None:
                latest_prediction_timestamp = '1970-01-01 00:00:00'  # Default if no predictions exist

            # Fetch new sensor data (only after the latest prediction timestamp)
            sensor_data = fetch_sensor_data(latest_prediction_timestamp)

            if not sensor_data.empty:
                # Extract features for prediction
                X_real_time = sensor_data[['temperature', 'vibration', 'current']]

                # Make predictions
                predictions = model.predict(X_real_time)
                probabilities = model.predict_proba(X_real_time)[:, 1]  # Probability of failure

                # Prepare data for insertion
                insert_data = [
                    (row.timestamp, row.motor_id, prob, pred, suggest_maintenance_tasks(prob))
                    for row, prob, pred in zip(sensor_data.itertuples(index=False), probabilities, predictions)
                ]

                # Store predictions and maintenance tasks in the database
                store_predictions(engine, insert_data)
                logging.info(f'Successfully inserted {len(insert_data)} predictions and maintenance tasks.')

            else:
                logging.info("No new sensor data available. Waiting for the next cycle.")

            # Sleep for a specified interval before the next iteration
            time.sleep(5)  # Fetch data every second

    except Exception as e:
        logging.error(f'Error occurred: {e}')

if __name__ == "__main__":
    main()
