import pandas as pd

# Input and output file paths
input_file = "pool-30.csv"
output_file = "pool-30.csv"

# Read CSV
df = pd.read_csv(input_file)

# Map 'Active' -> 1 and 'Inactive' -> 0
df['Label'] = df['Label'].map({'Active': 1, 'Inactive': 0})

# Save the updated file
df.to_csv(output_file, index=False)

print(f"Updated file saved to {output_file}")
