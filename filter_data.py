import pandas as pd

# --- File paths ---
large_file = "train-40.csv"
small_file = "subset-10.csv"
matched_file = "subset-10.csv"
remaining_file = "pool-30.csv"

# --- Read files ---
df_large = pd.read_csv(large_file)
df_small = pd.read_csv(small_file)

key_col = "PUBCHEM_CID"

# --- Matching rows (intersection) ---
matched_df = df_large[df_large[key_col].isin(df_small[key_col])]

# --- Remaining rows (not in small file) ---
remaining_df = df_large[~df_large[key_col].isin(df_small[key_col])]

# --- Save results ---
matched_df.to_csv(matched_file, index=False)
remaining_df.to_csv(remaining_file, index=False)

print(f"Matched rows saved to: {matched_file} ({len(matched_df)} rows)")
print(f"Remaining rows saved to: {remaining_file} ({len(remaining_df)} rows)")
