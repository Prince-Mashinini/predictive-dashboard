import matplotlib.pyplot as plt
import pandas as pd

# Data to display in the table
metrics_data = {
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)'],
    'Value': [0.0197, 0.0020, 0.0450, 0.9643]
}

# Create a DataFrame for the metrics
df_metrics = pd.DataFrame(metrics_data)

# Create a figure and axis to display the table
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust figure size as needed

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False) 
ax.set_frame_on(False)

# Create the table and display it
table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)  # Adjust scale as needed

# Save the figure as an image
plt.savefig('model_performance_metrics.png', bbox_inches='tight', dpi=300)

# Display the image
plt.show()
