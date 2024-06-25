import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
from helper_functions import hh_model

# Load model parameters from JSON file
path_to_model_file = 'model_parameters.json'
with open(path_to_model_file, 'r') as json_file:
    model_data = json.load(json_file)

# Access the parameters and confidence intervals
pKa_cf = model_data['parameters']['pKa']
tau_HA_cf = model_data['parameters']['tau_HA']
tau_Aminus_cf = model_data['parameters']['tau_Aminus']

# Generate an array of pH values
pH_values = np.linspace(5, 7.5, 1000)
# Calculate the corresponding lifetime values
lifetime_values = hh_model(pH_values, pKa_cf, tau_HA_cf, tau_Aminus_cf)

# Create the spline interpolation function
# there are 3 options how it is possible to interpolate the data 
# 1) do not allow extrapolation=False. This will turn all values outside the range to NAN
# 2) include clipping the data, this way it will keep them 7.5 and 5.0 if they fall in the edge (pretty much as raw data looks like)
# 3) dont clip or convert to nan (comment the line with np clip) and have funny values, that you need to filter

spline_interpolator = CubicSpline(lifetime_values, pH_values, extrapolate=True)

# Load the image data from CSV
image_data_path = Path(r"G:\SP8_FLIM\0_Workflow_OPTIM\pH\Outputs\master_table.csv")
image_data = pd.read_csv(image_data_path)

# Extract mean_tau values as numpy arrays
lifetimes_from_image_data = image_data['mean_tau'].to_numpy()

# Predict pH values for the new lifetimes using the spline interpolation function
predicted_pH_values = spline_interpolator(lifetimes_from_image_data)
predicted_pH_values = np.clip(predicted_pH_values, 5.0, 7.5)

# Visualize the original model and the spline interpolation
plt.plot(pH_values, lifetime_values, label='Model', color='blue')
plt.scatter(predicted_pH_values, lifetimes_from_image_data, color='green', label='Predicted pH from Spline Interpolation')
plt.xlabel('pH')
plt.ylabel('Average Lifetime')
plt.legend(loc='upper left')
plt.show()

# Add the pH values as a new column to the DataFrame 
image_data = image_data.assign(predicted_pH_values=predicted_pH_values)

# Save the DataFrame back to CSV with the new column
image_data.to_csv("master_table_with_ph.csv", index=False)
