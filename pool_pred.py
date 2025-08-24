import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, NNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.preprocessing import StandardScaler
from joblib import load
import tensorflow as tf
from rdkit import Chem

# ---------------------------
# Settings
# ---------------------------
baseline_models = {
    "attention": "pool-30-desc.csv",
    "CNN": "pool-30-desc.csv",
    "GCN": "pool-30.csv",
    "GNN_attention": "pool-30.csv"
}

baseline_folder_prefix = ""  # baseline folders: attention/, CNN/, etc.
pool_folder = ""             # folder where pool CSVs are stored

meta_model_file = "meta_model1/meta_CNN/model_meta_CNN.keras"
meta_scaler_file = "meta_model1/meta_CNN/scaler_meta_CNN.joblib"

output_dir = "pool_pred1"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Utility functions for GCN/GNN
# ---------------------------
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # undirected
        feat = bond_features(bond)
        edge_attr += [feat, feat]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


#---GCN-----
class GCNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super(GCNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)
    
class GNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_heads=4):
        super(GNNClassifier, self).__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=self.edge_net,
            aggr='mean'
        )
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Message passing
        x = self.nnconv(x, edge_index, edge_attr)
        x = F.relu(x)

        # Convert to dense batch: (batch_size, max_num_nodes, hidden_dim)
        x_dense, mask = to_dense_batch(x, batch)  # mask is (batch_size, max_num_nodes)

        # Apply MultiheadAttention (query, key, value are all x)
        attn_output, _ = self.multihead_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)

        # Aggregate the output: mean over node dimension (masked)
        attn_output[~mask] = 0  # mask out padded nodes
        graph_embeddings = attn_output.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (batch_size, hidden_dim)

        # Final MLP
        x = F.relu(self.lin1(graph_embeddings))
        return self.lin2(x).squeeze(1)

# ---------------------------
# Step 1: Predict baseline probabilities on pool
# ---------------------------
stacked_df = None

for model_name, pool_file_name in baseline_models.items():
    pool_path = os.path.join(pool_folder, pool_file_name)
    model_dir = os.path.join(baseline_folder_prefix, model_name)

    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"{pool_path} not found")

    df_pool = pd.read_csv(pool_path)
    pubchem_ids = df_pool['PUBCHEM_CID'].values

    # CNN / Attention: descriptor matrix
    if model_name in ["attention", "CNN"]:
        # Drop PUBCHEM_CID and Label (if exists)
        drop_cols = ['PUBCHEM_CID']
        if 'Label' in df_pool.columns:
            drop_cols.append('Label')
        X_pool = df_pool.drop(columns=drop_cols).values

        # Apply scaler if exists
        scaler_path = os.path.join(model_dir, f"scaler_{model_name}.joblib")
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            X_pool = scaler.transform(X_pool)

        # Load model
        model_path = os.path.join(model_dir, f"model_{model_name}.keras")
        model = tf.keras.models.load_model(model_path)

        # Expand dims for CNN/Attention
        if len(X_pool.shape) == 2:
            X_pool_exp = np.expand_dims(X_pool, axis=-1)

        # Predict probabilities
        y_prob = model.predict(X_pool_exp, verbose=0).flatten()
    
   # GCN / GNN_attention
    else:
        # Convert SMILES to graphs
        graphs = [mol_to_graph(s) for s in df_pool['SMILES'] if is_valid_molecule(s)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "GCN":
            node_dim = graphs[0].x.shape[1]
            gnn_model = GCNClassifier(node_dim).to(device)
        else:
            node_dim = graphs[0].x.shape[1]
            edge_dim = graphs[0].edge_attr.shape[1]
            gnn_model = GNNClassifier(node_dim, edge_dim).to(device)

        # Load state_dict
        model_path = os.path.join(model_dir, f"model_{model_name}.keras")
        gnn_model.load_state_dict(torch.load(model_path, map_location=device))
        gnn_model.eval()

        # Predict
        loader = DataLoader(graphs, batch_size=32)
        y_prob = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = gnn_model(batch)
                y_prob.extend(torch.sigmoid(out).cpu().numpy())

    # Create DataFrame for stacking
    df_probs = pd.DataFrame({
        'PUBCHEM_CID': df_pool['PUBCHEM_CID'].values[:len(y_prob)],
        f'y_prob_{model_name}': y_prob
    })

    # Merge into stacked dataframe
    if stacked_df is None:
        stacked_df = df_probs
    else:
        stacked_df = pd.merge(stacked_df, df_probs, on='PUBCHEM_CID', how='inner')

# Save stacked baseline features
stacked_file = os.path.join(output_dir, "stacked_pool_features.csv")
stacked_df.to_csv(stacked_file, index=False)
print(f"Stacked pool features saved to {stacked_file}")

# ---------------------------
# --- Meta-model prediction ---
# ---------------------------
# Load meta-model and scaler
meta_scaler = load(meta_scaler_file)
meta_model = tf.keras.models.load_model(meta_model_file)

X_pool_meta = stacked_df[[c for c in stacked_df.columns if c.startswith('y_prob_')]].values
X_pool_meta_scaled = meta_scaler.transform(X_pool_meta)

y_prob_meta = meta_model.predict(X_pool_meta_scaled, verbose=0).flatten()
y_pred_meta = (y_prob_meta >= 0.5).astype(int)

# Save probabilities
prob_file = os.path.join(output_dir, "pool_meta_prob.csv")
pool_prob_df = stacked_df[['PUBCHEM_CID']].copy()
pool_prob_df['y_prob_meta'] = y_prob_meta
pool_prob_df.to_csv(prob_file, index=False)
print(f"Meta-model probabilities saved to {prob_file}")

# Save predictions
pred_file = os.path.join(output_dir, "pool_meta_pred.csv")
pool_pred_df = stacked_df[['PUBCHEM_CID']].copy()
pool_pred_df['y_pred_meta'] = y_pred_meta
pool_pred_df.to_csv(pred_file, index=False)
print(f"Meta-model predictions saved to {pred_file}")
