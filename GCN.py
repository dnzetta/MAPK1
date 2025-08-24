import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split


# --- Atom and Bond Features ---
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


# --- Convert SMILES to PyG Data object ---
def mol_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # Undirected
        feat = bond_features(bond)
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


# --- Load and Preprocess Data ---
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    #df[label_col] = df[label_col].map({"Active": 1, "Inactive": 0})
    df = df[df[smiles_col].apply(is_valid_molecule)].reset_index(drop=True)
    df = df.dropna(subset=[smiles_col])
    return df

def split_data(output_dir, train_val_df, label_col):
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_col], random_state=42)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    return train_df, val_df

def convert_to_graphs(df, smiles_col, label_col):
    df['graph'] = df.apply(lambda row: mol_to_graph(row[smiles_col], row[label_col]), axis=1)
    return df['graph'].dropna().tolist()

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

# --- Metrics ---
def evaluate(model, loader, device):
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_logits.extend(logits.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = torch.sigmoid(torch.tensor(y_logits)).numpy()
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, auc, auprc, mcc, y_probs, y_pred


# --- Main ---
if __name__ == "__main__":
    name = "GCN"
    print(f"Model: {name}")
    print("📊 Loading and preprocessing data...\n")

    filename = 'data'
    file_path = "subset-10.csv"
    smiles_col = "SMILES"
    label_col = "Label"

    # Define output directory
    output_dir = f"{name}"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(file_path, smiles_col, label_col)
    print(f"Number of valid SMILES: {len(df)}")

    train_df, val_df = split_data(output_dir, df, label_col)
    test_df = pd.read_csv(os.path.join('test-10.csv'))

    train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
    val_graphs = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    print(f"📊 Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_graphs[0].x.shape[1]

    model = GCNClassifier(node_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_weight = torch.tensor([
        len(train_df[train_df[label_col] == 0]) / len(train_df[train_df[label_col] == 1])
    ], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("🧪 Start training...\n")
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)
        acc, auc, auprc, mcc, y_prob, y_pred = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, f"model_{name}.keras"))

    # Evaluate and save predictions
    print("\n📊 Final Test Set Evaluation")

    def predict_and_save(loader, df, set_name):
        acc, auc, auprc, mcc, y_prob, y_pred = evaluate(model, loader, device)
        cid_list = df["PUBCHEM_CID"].values

        prob_df = pd.DataFrame({
            "PUBCHEM_CID": cid_list,
            "y_prob": y_prob
        })
        prob_df.to_csv(os.path.join(output_dir, f"{set_name}_prob_{name}.csv"), index=False)

        pred_df = pd.DataFrame({
            "PUBCHEM_CID": cid_list,
            "y_pred": y_pred
        })
        pred_df.to_csv(os.path.join(output_dir, f"{set_name}_pred_{name}.csv"), index=False)
        return y_prob, y_pred

    # Save train, val, test predictions
    train_loader_all = DataLoader(train_graphs, batch_size=32)
    val_loader_all = DataLoader(val_graphs, batch_size=32)
    test_loader_all = DataLoader(test_graphs, batch_size=32)

    y_prob_train, y_pred_train = predict_and_save(train_loader_all, train_df, "train")
    y_prob_val, y_pred_val = predict_and_save(val_loader_all, val_df, "val")
    y_prob_test, y_pred_test = predict_and_save(test_loader_all, test_df, "test")

    # Final Test Evaluation
    acc, auc, auprc, mcc, y_probs, y_pred = evaluate(model, test_loader_all, device)
    print(f"[{name},{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f}]")

    result_row = {
        "Model": name,
        "Accuracy": round(acc, 3),
        "AUROC": round(auc, 3),
        "AUPRC": round(auprc, 3),
        "MCC": round(mcc, 3)
    }

    result_file = os.path.join(output_dir, "result.csv")
    try:
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC"])

    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)
