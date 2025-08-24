import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

def get_prob_0(name, df):
    # Check if the 'class_1_proba' column exists
    if 'y_prob_meta' in df.columns:
        # Calculate the class 0 probabilities
        df['y_pred_meta_0'] = 1 - df['y_prob_meta']

        # Save the updated DataFrame to a new CSV file
        df.to_csv(os.path.join(name, 'y_prob_pool_meta_binary.csv'), index=True)
        print("The updated CSV file  'y_prob_pool_meta_binary.csv' has been created successfully.")
    else:
        print("The column 'y_prob_pool_average' does not exist in the dataset.")

def entropy_sampling(name, df):
    proba = df.values
    # Calculate entropy for each row (sample)
    entropies = entropy(proba.T)  # scipy's entropy expects transposed input (shape = [n_classes, n_samples])
    df['y_prob_entropy'] = entropies  # Add entropy column
    df.to_csv(os.path.join(name, 'entropy_prob.csv'), index=True)
    print(f"Entropies have been calculated and saved.")

def entropy_sort(name, df, percentage=None, exact_number=None):
    # Sort compounds based on entropy in descending order
    df_sorted = df.sort_values(by='y_prob_entropy', ascending=False)
    
    # Determine number of rows to select
    if exact_number is not None:
        n = min(exact_number, len(df_sorted))  # Prevent selecting more than available
    elif percentage is not None:
        n = int(len(df_sorted) * percentage)
    else:
        raise ValueError("Either 'percentage' or 'exact_number' must be provided.")
    
    # Split the data into two sets
    entropy_subset = df_sorted.iloc[:n]
    remaining_data = df_sorted.iloc[n:]
    
    # Save datasets
    entropy_subset.to_csv(os.path.join(name, "entropy_subset.csv"), index=True)
    remaining_data.to_csv(os.path.join(name, "remaining_pool.csv"), index=True)
    
    print(f"Split completed: {n} rows saved in 'entropy_subset.csv' and "
          f"{len(df_sorted) - n} in 'remaining_pool.csv'.")
    
def split_y_pool(name, df, entropy_subset_df):
    entropy_y_pool = df[df["PUBCHEM_CID"].isin(entropy_subset_df["PUBCHEM_CID"])]
    remaining_y_pool = df[~df["PUBCHEM_CID"].isin(entropy_subset_df["PUBCHEM_CID"])]
    entropy_y_pool.to_csv(os.path.join(name, "entropy_subset_y_pool.csv"), index=False)
    remaining_y_pool.to_csv(os.path.join(name, "remaining_y_pool.csv"), index=False)
    print("y_pool split subset and remaining.")

def split_data(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list,remaining_list):
    """
    This function is to generate the best 100 compounds and remaining compounds from model predicted following with their canonical_smiles or ecfp into two CSV files.
    -----
    large_filepath= the filepath to your large_filename (also for the output filepath later on).
    large_filename= the file that contains all compounds with their canonical_smiles/ecfp.
    list_filepath= the filepath to your list_filename.
    list_filename= the file that contains list of 100 compounds with their affinities as result after predicted.
    filtered_list= the file that generated and consists of 100 compounds following with their canonical_smiles/ecfp.
    remaining_list= the file that generated and consists of remaining compounds following with their canonical_smiles/ecfp.
    """
    
    large_df = pd.read_csv(os.path.join(large_filepath, large_filename))        # Load the large data
    compound_list_df = pd.read_csv(os.path.join(list_filepath, list_filename))  # Load the compound list

    # Check if 'PUBCHEM_CID' columns are in both DataFrames
    if 'PUBCHEM_CID' not in large_df.columns:
        print(f"'PUBCHEM_CID' column not found in {large_filename}")
    if 'PUBCHEM_CID' not in compound_list_df.columns:
        print(f"'PUBCHEM_CID' column not found in {list_filename}")
    
    large_df['PUBCHEM_CID'] = large_df['PUBCHEM_CID'].astype(str).str.strip() # Standardize 'PUBCHEM_CID' data types and strip any leading/trailing whitespace
    compound_list_df['PUBCHEM_CID'] = compound_list_df['PUBCHEM_CID'].astype(str).str.strip()
    filtered_list_df = large_df[large_df['PUBCHEM_CID'].isin(compound_list_df['PUBCHEM_CID'])]    # Filter the DataFrame to include only the compounds in the compound list
    #remaining_list_df = large_df[~large_df['PUBCHEM_CID'].isin(compound_list_df['PUBCHEM_CID'])]  # Filter the ECFP DataFrame to include only the compounds NOT in the compound list
    
    # Save the filtered DataFrame to a new CSV file
    filtered_list_df.to_csv(os.path.join(output_path, filtered_list), index=False)

    # Save the remaining compounds to another CSV file
    #remaining_list_df.to_csv(os.path.join(output_path, remaining_list), index=False)


def merge_dataframes(file_paths, output_path, how='outer'):
    """
    Merge multiple CSV files into a single DataFrame and save it.
    
    Parameters:
    file_paths (list of str): List of file paths to CSV files to be merged.
    output_path (str): Path to save the merged DataFrame.
    how (str): Type of merge to be performed. Options are 'inner', 'outer', 'left', or 'right'. Default is 'outer'.
    """
    # Read and merge the DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    df_merged = pd.concat(dfs, axis=0, ignore_index=True, join=how)
    
    df_merged.to_csv(output_path, index=False)      # Save the merged DataFrame to CSV

def main():
    name = "entropy"
    os.makedirs(name, exist_ok=True)
    df = pd.read_csv(os.path.join('pool_pred1', 'pool_meta_prob.csv'), index_col=0)
    get_prob_0(name, df)

    df = pd.read_csv(os.path.join(name, "y_prob_pool_meta_binary.csv"), index_col=0)
    print(df.dtypes)
    entropy_sampling(name, df)
    entropy_cal_file = pd.read_csv(os.path.join(name, "entropy_prob.csv"), index_col=0)
    #entropy_sort(name, entropy_cal_file, percentage=0.05)
    entropy_sort(name, entropy_cal_file, exact_number=5)  # top 5 rows
    y_pool = pd.read_csv(os.path.join("pool-30.csv"))  #'pool', 
    entropy_subset_df = pd.read_csv(os.path.join(name, "entropy_subset.csv"))
    split_y_pool(name, y_pool, entropy_subset_df)

    large_filepath = ''    #/descriptor   /None
    large_filename = 'train-40.csv'
    list_filepath = name
    list_filename = 'entropy_subset.csv'
    output_path = name
    filtered_list = 'x_subset_0.05.csv'
    remaining_list = 'remaining_trainset.csv'
    split_data(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list,remaining_list)
    
    file_paths = [
        os.path.join('', 'subset-10.csv'), #/descriptor    /smiles    
        os.path.join(name,'x_subset_0.05.csv')
    ]
    merge_dataframes(file_paths, os.path.join(name,'x_subset.csv'))

    split_data(large_filepath, large_filename, list_filepath=name, list_filename='remaining_pool.csv', output_path=name, filtered_list='x_pool.csv',remaining_list='remaining.csv')
    
if __name__ == "__main__":
    main()
