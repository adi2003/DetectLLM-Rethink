"""
Binary classifier using a learned fusion function over two detector features.
Combines log likelihood and log rank for robust detection without entropy.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import json


class SigmoidEnsembleClassifier(nn.Module):
    """
    Learned fusion function over normalized [log_likelihood, log_rank].
    Output: probability of being machine-generated (1) vs human (0).

    The model uses a quadratic interaction score:
      z = b + w1*x1 + w2*x2 + w12*x1*x2 + w11*x1^2 + w22*x2^2
      p = sigmoid(z)
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.zeros(1))
        self.w12 = nn.Parameter(torch.zeros(1))
        self.w11 = nn.Parameter(torch.zeros(1))
        self.w22 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the learned ll-rank fusion score."""
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        logits = (
            self.bias
            + self.w1 * x1
            + self.w2 * x2
            + self.w12 * (x1 * x2)
            + self.w11 * (x1 * x1)
            + self.w22 * (x2 * x2)
        )
        return torch.sigmoid(logits)


class EnsembleTrainer:
    """Trains and evaluates the ll-rank fusion classifier."""
    
    def __init__(self, learning_rate: float = 0.001, epochs: int = 500, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SigmoidEnsembleClassifier().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.epochs = epochs
        self.train_losses = []
        self.normalization_stats = []  # Store min/max for feature normalization in feature order
    
    def prepare_features(self, 
                        log_likelihoods: List[float], 
                        log_ranks: List[float], 
                        labels: List[int] = None,
                        fit_normalization: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare and normalize ll/rank features to [0, 1]."""
        # Stack features
        features = np.column_stack([log_likelihoods, log_ranks])
        
        # Normalize each feature to [0, 1] range using min-max normalization
        features_normalized = np.zeros_like(features, dtype=np.float32)
        
        for i in range(features.shape[1]):
            col = features[:, i]
            if fit_normalization:
                if i == 0:
                    self.normalization_stats = []
                col_min = np.min(col)
                col_max = np.max(col)
                self.normalization_stats.append({'min': float(col_min), 'max': float(col_max)})
            else:
                col_min = self.normalization_stats[i]['min']
                col_max = self.normalization_stats[i]['max']
            
            if col_max - col_min > 1e-6:
                features_normalized[:, i] = (col - col_min) / (col_max - col_min)
            else:
                features_normalized[:, i] = col
        
        # Convert to tensors
        X = torch.tensor(features_normalized, dtype=torch.float32).to(self.device)
        
        if labels is not None:
            y = torch.tensor(np.array(labels, dtype=np.float32).reshape(-1, 1), dtype=torch.float32).to(self.device)
            return X, y
        else:
            return X, None
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model."""
        self.model.train()
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            self.train_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.cpu().numpy().flatten()
    
    def evaluate(self, X: torch.Tensor, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        roc_auc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'predictions': y_pred
        }
    
    def save(self, model_path: str, stats_path: str = None) -> None:
        """Save model and normalization statistics."""
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        if stats_path is None:
            stats_path = model_path.replace('.pt', '_stats.json')
        
        with open(stats_path, 'w') as f:
            json.dump(self.normalization_stats, f, indent=2)
        print(f"Normalization stats saved to {stats_path}")
    
    def load(self, model_path: str, stats_path: str = None, device: str = 'cuda') -> None:
        """Load model and normalization statistics."""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SigmoidEnsembleClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
        
        if stats_path is None:
            stats_path = model_path.replace('.pt', '_stats.json')
        
        with open(stats_path, 'r') as f:
            self.normalization_stats = json.load(f)
        print(f"Normalization stats loaded from {stats_path}")

    def get_fusion_formula(self) -> str:
        """Return the learned normalized-space fusion function."""
        b = float(self.model.bias.detach().cpu().item())
        w1 = float(self.model.w1.detach().cpu().item())
        w2 = float(self.model.w2.detach().cpu().item())
        w12 = float(self.model.w12.detach().cpu().item())
        w11 = float(self.model.w11.detach().cpu().item())
        w22 = float(self.model.w22.detach().cpu().item())
        return (
            "score = sigmoid("
            f"{b:.6f} + "
            f"{w1:.6f}*ll_norm + "
            f"{w2:.6f}*logrank_norm + "
            f"{w12:.6f}*ll_norm*logrank_norm + "
            f"{w11:.6f}*ll_norm^2 + "
            f"{w22:.6f}*logrank_norm^2"
            ")"
        )
