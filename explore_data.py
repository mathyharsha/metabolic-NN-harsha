import pandas as pd

datafile = "./data/2025-06-23_full_training_data_149261_samples.csv"

df = pd.read_csv(datafile)

mean = df['ME1_flux'].mean()
std = df['ME1_flux'].std()

print(f"ME1_flux mean: {mean}")
print(f"ME1_flux standard deviation: {std}")

print(df['ME1_flux'])

def preprocess_zero_inflated(y_col, epsilon=1e-6):
    is_zero = (y_col.abs() < epsilon)
    return {
        'zero_mask': is_zero,
        'nonzero_values': y_col[~is_zero],
        'nonzero_ratio': 1 - is_zero.mean()
    }

me1_stats = preprocess_zero_inflated(df['ME1_flux'])
print(f"Non-zero ratio: {me1_stats['nonzero_ratio']:.4f}")

mean = df['EX_for_e_flux'].mean()
std = df['EX_for_e_flux'].std()

print(f"EX_for_e_flux mean: {mean}")
print(f"EX_for_e_flux standard deviation: {std}")
