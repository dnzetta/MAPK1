from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd

def calculate_descriptors(df, smiles_col):
    """
    Compute molecular descriptors using RDKit.
    ------
    df: DataFrame
    smiles_col: Column name containing SMILES strings
    """
    # Define the list of descriptors to calculate
    descriptor_functions = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'TPSA': rdMolDescriptors.CalcTPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles,
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
        'RingCount': rdMolDescriptors.CalcNumRings,
        'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms,
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles,
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
        'NumValenceElectrons': Descriptors.NumValenceElectrons,
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles,
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds,
         # Add more descriptors as needed
    }

    def get_descriptors(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            return [func(mol) for func in descriptor_functions.values()]
        except:
            return [None] * len(descriptor_functions)

    descriptors_df = df[smiles_col].apply(get_descriptors).apply(pd.Series)
    descriptors_df.columns = list(descriptor_functions.keys())
    return descriptors_df

# Example usage
if __name__ == "__main__":
    # Sample DataFrame
    df = pd.read_csv('smiles-50.csv')
    
    # Calculate RDKit fingerprints
    descriptors = calculate_descriptors(df, 'SMILES')
    
    # Add PUBCHEM_CID as the first column and Label in the last column
    descriptors.insert(0, 'PUBCHEM_CID', df['PUBCHEM_CID'])
    descriptors['Label'] = df['Label'].map({'Active': 1, 'Inactive': 0})
    # Save the resulting DataFrame to a CSV file
    output_file = 'descriptors.csv'
    descriptors.to_csv(output_file, index=False)

    # Display the result
    print(descriptors)