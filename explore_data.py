import pandas as pd

datafile = "./data/2025-06-23_full_training_data_149977_samples.csv"

df = pd.read_csv(datafile)

mean = df['ME1_flux'].mean()
std = df['ME1_flux'].std()

print(f"ME1_flux mean: {mean}")
print(f"ME1_flux standard deviation: {std}")

print(df['ME1_flux'])
