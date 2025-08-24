import pandas as pd
import os

# List of baseline folders (each folder name will be used as model_name)
baseline_folders = ["attention", "CNN", "GCN", "GNN_attention"]

# Paths for the label files
train_label_file = "subset-10.csv"  # should contain PUBCHEM_CID,Label
test_label_file = "test-10.csv"

# Output folder
output_dir = "meta_model1"
os.makedirs(output_dir, exist_ok=True)

def stack_prob_files(file_type="train_prob"):
    """
    Stack probability files (train or test) side by side based on PUBCHEM_CID.
    """
    merged_df = None
    
    for folder in baseline_folders:
        model_name = folder
        file_path = os.path.join(folder, f"{file_type}_{model_name}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        df = pd.read_csv(file_path)
        
        # Standardize probability column to y_prob_{model_name}
        if 'Probability' in df.columns:
            df = df[['PUBCHEM_CID', 'Probability']].rename(columns={'Probability': f'y_prob_{model_name}'})
        elif 'y_prob' in df.columns:
            df = df[['PUBCHEM_CID', 'y_prob']].rename(columns={'y_prob': f'y_prob_{model_name}'})
        else:
            raise ValueError(f"No probability column found in {file_path}")
        
        # Merge with final DataFrame
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='PUBCHEM_CID', how='inner')
    
    return merged_df

# Stack train probabilities
train_merged = stack_prob_files("train_prob")

# Merge train labels
train_labels = pd.read_csv(train_label_file)
train_merged = pd.merge(train_merged, train_labels[['PUBCHEM_CID', 'Label']], on='PUBCHEM_CID', how='left')

train_merged.to_csv(os.path.join(output_dir, "stacked_train_prob.csv"), index=False)

# Stack test probabilities
test_merged = stack_prob_files("test_prob")

# Merge test labels
test_labels = pd.read_csv(test_label_file)
test_merged = pd.merge(test_merged, test_labels[['PUBCHEM_CID', 'Label']], on='PUBCHEM_CID', how='left')

test_merged.to_csv(os.path.join(output_dir, "stacked_test_prob.csv"), index=False)

print(f"Stacked train and test probability files with labels saved in {output_dir}")