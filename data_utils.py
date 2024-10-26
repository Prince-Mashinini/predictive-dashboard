import pandas as pd
from sqlalchemy import create_engine

# Database connection using SQLAlchemy
def all_data():
    engine = create_engine('postgresql://motor_user:Admin@localhost:5432/motor_db')
    
    # Fetch predictions and acknowledgment status from the database
    query_predictions = """
        SELECT timestamp, motor_id, failure_probability, maintenance_tasks, acknowledged
        FROM motor_failure_predictions
        ORDER BY timestamp ASC
    """
    df_predictions = pd.read_sql(query_predictions, engine)
    
    # Fetch sensor data (temperature, vibration, current)
    query_sensor_data = """
        SELECT timestamp, motor_id, temperature, vibration, current
        FROM motor_sensor_data
        ORDER BY timestamp ASC
    """
    df_sensor = pd.read_sql(query_sensor_data, engine)
    
    # Merge predictions with sensor data on timestamp and motor_id
    df_combined = pd.merge(df_predictions, df_sensor, on=["timestamp", "motor_id"], how="inner")

    # Ensure acknowledgment column exists in combined data
    if 'acknowledged' not in df_combined.columns:
        df_combined['acknowledged'] = False
    
    return df_combined