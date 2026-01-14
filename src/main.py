import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from model import InstitutionalDRL
import numpy as np

def generate_mock_market_data(assets=5, days=2000):
    # Simulates asset returns (PGIM/Fidelity style data)
    return torch.randn(days, assets)

def train_model():
    # 1. Setup Data
    market_data = generate_mock_market_data()
    # Dummy rewards for demonstration: target Sharpe Ratio maximization
    rewards = torch.randn(2000, 1) 
    dataset = TensorDataset(market_data, rewards)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # 2. Initialize Model
    model = InstitutionalDRL(input_dim=5, action_dim=5)

    # 3. Scalable Trainer (Supports multi-GPU out of the box)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto", 
        devices=1,
        precision="16-mixed" # Memory efficiency
    )

    print("Starting Institutional Portfolio Optimization Training...")
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    train_model()
