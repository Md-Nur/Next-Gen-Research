import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Dataset Class to construct sliding time windows
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size=14):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # Extract past window_size days of features
        X_window = self.X[idx : idx + self.window_size]
        # Target corresponds to the day immediately following the window
        y_target = self.y[idx + self.window_size]
        return X_window, y_target

# 2. Spatial-Temporal Attention Mechanism Layer
class SpatialTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, window_size):
        super(SpatialTemporalAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1) # *2 for Bidirectional
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out shape: [batch_size, window_size, hidden_dim * 2]
        scores = self.attention_weights(lstm_out) # [batch_size, window_size, 1]
        weights = self.softmax(scores) # Compute alignment probabilities
        
        # Compute context vector via weighted sum across the temporal dimension
        context = torch.sum(lstm_out * weights, dim=1) # [batch_size, hidden_dim * 2]
        return context, weights

# 3. Core STA-BiLSTM Architecture
class STABiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, output_dim=1):
        super(STABiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attention = SpatialTemporalAttention(hidden_dim, window_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x) # [batch_size, window_size, hidden_dim * 2]
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        return out, attn_weights

# 4. Pipeline Executive Logic
def train_and_evaluate_deep_model(data_path, target_col, window_size=14):
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    features = ["GHI_Bias_Corrected", "ClearSky_GHI", "Ambient_Temp", "Humidity", "Wind_Speed"]
    
    # We aggregate performance across our forward-chaining evaluation folds
    all_r2, all_mape = [], []
    
    print(f"\n==========================================")
    # Correct string manipulation for upper case output display
    print(f" Training STA-BiLSTM for: {target_col.upper()}")
    print(f"==========================================")
    
    for val_fold in range(5, 10): # Evaluating across the latter half stable folds
        train_df = df[df["Fold"] < val_fold]
        val_df = df[df["Fold"] == val_fold]
        
        # Scaling inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_val = scaler.transform(val_df[features])
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        
        # Build sliding window datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
        val_dataset = TimeSeriesDataset(X_val, y_val, window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Instantiate architecture
        model = STABiLSTM(input_dim=len(features), hidden_dim=64, window_size=window_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # Standard training loop execution
        model.train()
        for epoch in range(40):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
        # Validation Evaluation
        model.eval()
        predictions, ground_truth = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs, _ = model(batch_X)
                predictions.extend(outputs.squeeze().numpy())
                ground_truth.extend(batch_y.numpy())
                
        r2 = r2_score(ground_truth, predictions)
        mape = mean_absolute_percentage_error(np.array(ground_truth)+1e-5, np.array(predictions)+1e-5) * 100
        all_r2.append(r2)
        all_mape.append(mape)
        
    print(f"🚀 Final Result for {target_col} -> Mean R²: {np.mean(all_r2):.4f} | Mean MAPE: {np.mean(all_mape):.2f}%")

if __name__ == "__main__":
    # Test the model against the difficult Biogas variables first to verify the temporal fix
    train_and_evaluate_deep_model("data/processed_dataset.csv", "PlantA_E_biogas")
    train_and_evaluate_deep_model("data/processed_dataset.csv", "PlantB_E_biogas")