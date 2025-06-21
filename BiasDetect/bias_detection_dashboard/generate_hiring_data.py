import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate sample data
n_samples = 1000
ages = np.random.randint(20, 60, n_samples)
# Bin Age into categories: 20-30, 31-40, 41-50, 51-60
age_bins = pd.cut(ages, bins=[20, 30, 40, 50, 60], labels=['20-30', '31-40', '41-50', '51-60'], include_lowest=True)

data = {
    'name': [f"Person_{i}" for i in range(n_samples)],
    'email': [f"person{i}@example.com" for i in range(n_samples)],
    'phone': [f"555-555-{str(i).zfill(4)}" for i in range(n_samples)],
    'Age': age_bins,
    'Gender': random.choices(['Male', 'Female', 'Other'], weights=[0.5, 0.4, 0.1], k=n_samples),  # Slightly imbalanced
    'Race': random.choices(['White', 'Black', 'Asian', 'Hispanic', 'Other'], weights=[0.4, 0.2, 0.2, 0.15, 0.05], k=n_samples),  # Imbalanced
    'Department': random.choices(['IT', 'HR', 'Sales', 'Marketing'], k=n_samples),
    'DailyRate': np.random.normal(1000, 200, n_samples).astype(int),  # Normal distribution
    'MonthlyIncome': np.random.normal(6000, 1500, n_samples).astype(int),  # Normal distribution
    'YearsAtCompany': np.random.randint(0, 20, n_samples),
    'shortlisted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Imbalanced target
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce some missing values (5% of the data)
for col in ['DailyRate', 'MonthlyIncome']:
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, col] = np.nan

# Save to CSV
df.to_csv('hiring_data.csv', index=False)
print("Sample dataset 'hiring_data.csv' generated successfully.")