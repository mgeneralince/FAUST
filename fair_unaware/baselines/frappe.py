import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from ..neural_losses import diff_mmd_1d

class PostHocModule(nn.Module):
    """The T(x) network proposed by FRAPPÉ."""
    def __init__(self, input_dim, hidden_dims=(64,)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
class FrappeRegressor(BaseEstimator, RegressorMixin):
    """
    FRAPPÉ Post-Processing Framework adapted for continuous DP regression.
    Minimizes: MSE(Y_fair, Y_base) + lambda * DP_Penalty(Y_fair, S)
    """
    def __init__(
        self, 
        base_regressor, 
        hidden_dims=(64,), 
        lam=1.0, 
        epochs=100, 
        lr=0.01, 
        batch_size=None,
        mmd_sigma=0.1,
        random_state=42
    ):
        self.base_regressor = base_regressor
        self.hidden_dims = hidden_dims
        self.lam = float(lam)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mmd_sigma = mmd_sigma
        self.random_state = random_state

    def fit(self, X, S, y, X_unlab=None, S_unlab=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        self.base_regressor_ = clone(self.base_regressor)
        self.base_regressor_.fit(X, y)

        if X_unlab is None:
            X_unlab, S_unlab = X, S

        y_base = self.base_regressor_.predict(X_unlab)

        X_t = torch.tensor(X_unlab, dtype=torch.float32)
        S_t = torch.tensor(np.asarray(S_unlab).ravel(), dtype=torch.float32)
        y_base_t = torch.tensor(y_base, dtype=torch.float32)

        # Initialize T(x) 
        self.posthoc_ = PostHocModule(input_dim=X.shape[1], hidden_dims=self.hidden_dims)
        optimizer = optim.Adam(self.posthoc_.parameters(), lr=self.lr)

        dataset_size = X_t.shape[0]
        bs = self.batch_size if self.batch_size is not None else dataset_size

        # Train 
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, bs):
                indices = permutation[i:i+bs]
                batch_X = X_t[indices]
                batch_S = S_t[indices]
                batch_y_base = y_base_t[indices]

                optimizer.zero_grad()

                # T(X)
                delta = self.posthoc_(batch_X)
                # Y_fair = Y_base + T(X)
                y_fair = batch_y_base + delta

                #Divergence MSE(Y_fir, Y_base)
                loss_pred = torch.mean(delta ** 2)

                # mask1 = (batch_S == 1.0).float()
                # mask0 = (batch_S == 0.0).float()
                
                # mean1 = torch.sum(y_fair * mask1) / (torch.sum(mask1) + 1e-8)
                # mean0 = torch.sum(y_fair * mask0) / (torch.sum(mask0) + 1e-8)
                # loss_fair = (mean1 - mean0) ** 2

                # y1 = y_fair[batch_S == 1.0]
                # y0 = y_fair[batch_S == 0.0]
                # loss_fair = diff_mmd_1d(y1, y0, sigma=self.mmd_sigma)

                #MMD across all pairs 
                unique_groups = torch.unique(batch_S)
                loss_fair = torch.tensor(0.0, device=y_fair.device)
                if len(unique_groups) > 1:
                    for g1_idx in range(len(unique_groups)):
                        for g2_idx in range(g1_idx + 1, len(unique_groups)):
                            y1 = y_fair[batch_S == unique_groups[g1_idx]]
                            y2 = y_fair[batch_S == unique_groups[g2_idx]]
                            loss_fair += diff_mmd_1d(y1, y2, sigma=self.mmd_sigma)

                loss = loss_pred + self.lam * loss_fair

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        #Base 
        y_base = self.base_regressor_.predict(X)
        
        # correction (Unaware of S)
        self.posthoc_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            delta = self.posthoc_(X_t).numpy()
            
        return y_base + delta