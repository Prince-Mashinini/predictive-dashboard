import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import create_lagged_features, all_data

# Load the trained model
model = joblib.load("xgboost_failure_model.joblib")

# Fetch all the data for testing
df = all_data()

# Define the number of lags (consistent with training)
n_lags = 5

# Drop non-numeric columns (e.g., 'maintenance_tasks', 'timestamp')
df = df.drop(['maintenance_tasks', 'timestamp'], axis=1)

# Create lagged features
df_lagged = create_lagged_features(df, 'failure_probability', n_lags)

# Split data into features (X) and target (y)
X = df_lagged.drop(columns=['failure_probability'])
y = df_lagged['failure_probability']

# Make predictions using the trained model
predictions = model.predict(X)

# Calculate performance metrics
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Plotting
plt.figure(figsize=(15, 6))

# 1. Predicted vs Actual values plot
plt.subplot(1, 2, 1)
plt.plot(y, label='Actual')
plt.plot(predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Failure Probability')
plt.xlabel('Index')
plt.ylabel('Failure Probability')
plt.legend()

# Save the first plot
plt.savefig('actual_vs_predicted.png')

# 2. Residuals Plot (Actual - Predicted)
residuals = y - predictions
plt.subplot(1, 2, 2)
plt.scatter(y, residuals, color='red', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual Failure Probability')
plt.ylabel('Residuals')

# Save the residuals plot
plt.savefig('residual_plot.png')

# Histogram of residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')

# Save the histogram of residuals
plt.savefig('residuals_histogram.png')

print("Plots have been saved as 'actual_vs_predicted.png', 'residual_plot.png', and 'residuals_histogram.png'.")
