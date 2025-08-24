import os
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Attention
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# === Model Definitions ===
def attention_model(fingerprint_length):
    input_layer = Input(shape=(fingerprint_length,))
    dense_layer = Dense(64, activation='relu')(input_layer)
    reshape_layer = Reshape((1, 64))(dense_layer)
    attention_layer = Attention(use_scale=True)([reshape_layer, reshape_layer])
    attention_output = Reshape((64,))(attention_layer)
    output_layer = Dense(1, activation='sigmoid')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Utility Functions ===
def split_data(output_dir, train_val_df, label_col):
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_col], random_state=42)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    return train_df, val_df

def evaluate_model(model, x, y_true):
    y_prob = model.predict(x)
    y_pred = (y_prob > 0.5).astype(int)
    print(y_prob.shape)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, auc, auprc, mcc, y_prob, y_pred

# === Main Script ===
if __name__ == "__main__":
    name = "meta_attention"
    filename = "meta_model1"
    print(f"Model: {name}")
    file_path = f"{filename}/stacked_train_prob.csv"

    # Define output directory
    output_dir = f"{filename}/{name}"
    os.makedirs(output_dir, exist_ok=True)

    print("📊 Loading and preprocessing data...\n")
    df = pd.read_csv(file_path)

    features = df.drop(columns=['Label', 'PUBCHEM_CID'])
    labels = df['Label']

    print(labels)
    print(features)

    train_df, val_df = split_data(output_dir, df, label_col='Label')
    test_df = pd.read_csv(os.path.join(filename, 'stacked_test_prob.csv'))

    X_train = train_df.drop(columns=['Label', 'PUBCHEM_CID'])
    y_train = train_df['Label']
    X_val = val_df.drop(columns=['Label', 'PUBCHEM_CID'])
    y_val = val_df['Label']
    X_test = test_df.drop(columns=['Label', 'PUBCHEM_CID'])
    y_test = test_df['Label']

    print(X_train)
    print(y_train)

    X_train_np = np.array(X_train)
    X_val_np = np.array(X_val)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)

    from joblib import dump
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_np)
    dump(scaler, os.path.join(output_dir, f'scaler_{name}.joblib'))
    X_val = scaler.transform(X_val_np)
    X_test = scaler.transform(X_test_np)

    fingerprint_length = X_train.shape[1]
    model = attention_model(fingerprint_length)

    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    print("🧪 Start training...\n")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    model.save(os.path.join(output_dir, f"model_{name}.keras"))

    acc, auc, auprc, mcc, y_prob_train, y_pred_train = evaluate_model(model, X_train, y_train)
    train_probs = y_prob_train.flatten()  # in case it's (n,1)
    train_result_df = pd.DataFrame({
        'PUBCHEM_CID': train_df['PUBCHEM_CID'].values,
        'Probability': train_probs
    })
    train_result_df.to_csv(os.path.join(output_dir, f'train_prob_{name}.csv'), index=False)

    train_pred = y_pred_train.flatten()
    train_pred_df = pd.DataFrame({
        'PUBCHEM_CID': train_df['PUBCHEM_CID'].values,
        'Probability': train_pred
    })
    train_pred_df.to_csv(os.path.join(output_dir, f'train_pred_{name}.csv'), index=False)

    acc, auc, auprc, mcc, y_prob_val, y_pred_val = evaluate_model(model, X_val, y_val)
    val_probs = y_prob_val.flatten() # in case it's (n,1)
    val_probs_df = pd.DataFrame({
        'PUBCHEM_CID': val_df['PUBCHEM_CID'].values,
        'Probability': val_probs
    })
    val_probs_df.to_csv(os.path.join(output_dir, f'val_prob_{name}.csv'), index=False)

    val_pred = y_pred_val.flatten()
    val_pred_df = pd.DataFrame({
        'PUBCHEM_CID': val_df['PUBCHEM_CID'].values,
        'Probability': val_pred
    })
    val_pred_df.to_csv(os.path.join(output_dir, f'val_pred_{name}.csv'), index=False)

    print("\n📊 Final Test Set Evaluation")
    acc, auc, auprc, mcc, y_prob_test, y_pred_test = evaluate_model(model, X_test, y_test)
    test_probs = y_prob_test.flatten()  # in case it's (n,1)
    test_result_df = pd.DataFrame({
        'PUBCHEM_CID': test_df['PUBCHEM_CID'].values,
        'Probability': test_probs
    })
    test_result_df.to_csv(os.path.join(output_dir, f'test_prob_{name}.csv'), index=False)

    test_pred = y_pred_test.flatten()
    test_pred_df = pd.DataFrame({
        'PUBCHEM_CID': test_df['PUBCHEM_CID'].values,
        'Probability': test_pred
    })
    test_pred_df.to_csv(os.path.join(output_dir, f'test_pred_{name}.csv'), index=False)
    #acc, auc, auprc, mcc, y_prob = evaluate_model(model, X_test, y_test)
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
