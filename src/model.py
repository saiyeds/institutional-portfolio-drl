import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class InstitutionalDRL(pl.LightningModule):
    def __init__(self, input_dim=5, action_dim=5, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature Extractor (Shared)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor: Outputs Portfolio Weights (Must sum to 1.0)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: Estimates State Value (Risk-Adjusted Reward)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone(x)
        weights = self.actor(features)
        value = self.critic(features)
        return weights, value

    def training_step(self, batch, batch_idx):
        state, reward = batch
        weights, value = self(state)
        
        # Loss = Negative Log Likelihood * Advantage + MSE Value Loss
        # Designed for institutional stability
        advantage = reward - value.detach()
        loss_actor = -(torch.log(weights).mean() * advantage)
        loss_critic = F.mse_loss(value, reward)
        
        total_loss = loss_actor + loss_critic
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
