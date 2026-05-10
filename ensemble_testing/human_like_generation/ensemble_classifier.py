"""
Binary classifier using sigmoid activation to combine multiple detection features.
Combines log likelihood, log rank, and entropy for robust detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os


class SigmoidEnsembleClassifier(nn.Module):
    """
    Binary classifier using sigmoid activation.
    Inputs: [log_likelihood, log_rank, entropy]
    Output: probability of being machine-generated (1) vs human (0)
    """
    
    def __init__(self, input_dim: int = 3):
        super().__init__()
        # Dense layer with sigmoid output
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid output for binary classification."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class EnsembleTrainer:
    """Trains and evaluates the ensemble classifier."""
    
    def __init__(self, learning_rate: float = 0.001, epochs: int = 100, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SigmoidEnsembleClassifier(input_dim=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.epochs = epochs
        self.train_losses = []
        self.normalization_stats = {}  # Store min/max for feature normalization
    
    def prepare_features(self, 
                        log_likelihoods: List[float], 
                        log_ranks: List[float], 
                        entropies: List[float],
                        labels: List[int] = None,
                        fit_normalization: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare and normalize features."""
        # Stack features
        features = np.column_stack([log_likelihoods, log_ranks, entropies])
        
        # Normalize each feature to [0, 1] range using min-max normalization
        features_normalized = np.zeros_like(features, dtype=np.float32)
        
        for i in range(features.shape[1]):
            col = features[:, i]
            if fit_normalization:
                col_min = np.min(col)
                col_max = np.max(col)
                self.normalization_stats[i] = {'min': col_min, 'max': col_max}
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
            stats_path = model_path.replace('.pt', '_stats.pt')
        
        torch.save(self.normalization_stats, stats_path)
        print(f"Normalization stats saved to {stats_path}")
    
    def load(self, model_path: str, stats_path: str = None, device: str = 'cuda') -> None:
        """Load model and normalization statistics."""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SigmoidEnsembleClassifier(input_dim=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
        
        if stats_path is None:
            stats_path = model_path.replace('.pt', '_stats.pt')
        
        self.normalization_stats = torch.load(stats_path, map_location='cpu')
        print(f"Normalization stats loaded from {stats_path}")
