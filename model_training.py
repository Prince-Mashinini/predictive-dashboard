import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from joblib import dump
from data_utils import all_data
import joblib

# Function to create lagged features for time series
def create_lagged_features(df, target_column, n_lags=5):
    """
    Create lagged features for time series data.
    
    Args:
    df (pd.DataFrame): The input data frame with a timestamp index and feature columns.
    target_column (str): The target column to create lag features for.
    n_lags (int): The number of lagged features to create.
    
    Returns:
    pd.DataFrame: Dataframe with lagged features.
    """
    df_lagged = df.copy()
    
    # Create lag features for the target column
    for lag in range(1, n_lags + 1):
        df_lagged[f'{target_column}_lag_{lag}'] = df_lagged[target_column].shift(lag)
    
    # Drop rows with NaN values created due to lagging
    df_lagged = df_lagged.dropna()
    
    return df_lagged

def train_xgboost_model(df, target_column, n_lags=5):
    # Remove non-numeric columns (e.g., 'maintenance_tasks')
    df = df.drop(['maintenance_tasks', 'timestamp'], axis=1)  # Drop 'timestamp' and 'maintenance_tasks' columns

    # Create lagged features
    df_lagged = create_lagged_features(df, target_column, n_lags)
    print()
    
    # Define the features (X) and the target (y)
    X = df_lagged.drop(columns=[target_column])
    y = df_lagged[target_column]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, 'xgboost_failure_model.joblib')
    print("Model saved to 'xgboost_failure_model.joblib'")
    
    return model


def forecast_future_failures_xgboost(df, model, forecast_period_seconds=60, n_lags=5):
    """
    Forecast future failure probabilities using the XGBoost model.
    """
    # Ensure the DataFrame is not empty
    if df.empty:
        print("Error: DataFrame is empty. Cannot perform forecasting.")
        return pd.DataFrame()

    # Ensure the DataFrame has enough rows for lagging
    if len(df) < n_lags:
        print(f"Error: Not enough data to create lagged features. DataFrame has {len(df)} rows, but requires at least {n_lags}.")
        return pd.DataFrame()

    # Preserve the timestamp for generating future timestamps later
    timestamps = df['timestamp'].copy()  # Keep a copy of the timestamp column

    # Drop non-numeric columns (e.g., 'maintenance_tasks', 'timestamp')
    df = df.drop(['maintenance_tasks', 'timestamp'], axis=1, errors='ignore')  # Drop 'timestamp' and 'maintenance_tasks'

    # Create lagged features
    df_lagged = create_lagged_features(df, 'failure_probability', n_lags=n_lags)

    # Ensure the lagged DataFrame is not empty after processing
    if df_lagged.empty:
        print("Error: Lagged DataFrame is empty. Cannot perform forecasting.")
        return pd.DataFrame()

    # Handle any NaN values created by the lagging process
    df_lagged = df_lagged.dropna()

    if df_lagged.empty:
        print("Error: Lagged DataFrame is empty after dropping NaN values. Cannot perform forecasting.")
        return pd.DataFrame()

    # Check if df_lagged has at least one row to avoid out-of-bounds access
    if len(df_lagged) == 0:
        print("Error: No rows in the lagged DataFrame after dropping NaNs.")
        return pd.DataFrame()

    # Extract the latest row for forecasting
    try:
        latest_features = df_lagged.iloc[-1].drop(['failure_probability'])
    except KeyError as e:
        print(f"KeyError: {e}. The 'failure_probability' column may be missing.")
        return pd.DataFrame()

    # Prepare input for prediction
    input_data = np.array(latest_features).reshape(1, -1)

    # Forecast future probabilities for the next 'forecast_period_seconds'
    forecast = []
    for i in range(forecast_period_seconds):
        future_prob = model.predict(input_data)[0]  # Predict next failure probability
        forecast.append(future_prob)

        # Update the input data with the new prediction for rolling forecasting
        input_data = np.roll(input_data, -1)  # Shift data
        input_data[-1] = future_prob  # Append the new prediction

    # Generate future timestamps (1 second intervals) using the preserved timestamps
    last_timestamp = timestamps.max()  # Use the original timestamps
    future_timestamps = [last_timestamp + timedelta(seconds=i + 1) for i in range(forecast_period_seconds)]

    # Create a DataFrame with forecasted timestamps and failure probabilities
    forecast_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'failure_probability': forecast
    })

    return forecast_df


# Function to load the trained XGBoost model
def load_xgboost_model(model_path='xgboost_failure_model.joblib'):
    """
    Load a pre-trained XGBoost model from file.
    
    Args:
    model_path (str): Path to the saved XGBoost model.
    
    Returns:
    xgb.XGBRegressor: Loaded XGBoost model.
    """
    from joblib import load
    return load(model_path)

# Train and save the model when the script is run
if __name__ == "__main__":
    # Assuming you have a dataframe `df` with failure probabilities
    df = all_data()  # Load your data here

    # Train the model
    train_xgboost_model(df, 'failure_probability', n_lags=5)
