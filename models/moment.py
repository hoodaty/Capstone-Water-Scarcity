import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from momentfm import MOMENTPipeline

class MOMENTQuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Wrapper to train Lower, Median, and Upper quantile models with CQR calibration,
    utilizing Hugging Face's MOMENT Foundation Model for representation learning.
    """
    def __init__(self, alpha=0.1, model_name="AutonLab/MOMENT-1-base", 
                 epochs=50, lr=1e-3, batch_size=32, device=None, **kwargs):
        self.alpha = alpha
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        self.q_score = 0.0
        
        self.moment = None
        self.head = None

    def _pinball_loss(self, y_true, y_pred, quantile):
        err = y_true - y_pred
        return torch.mean(torch.max(quantile * err, (quantile - 1.0) * err))

    def fit(self, X, y):
        # MOMENT expects shape: (Batch_size, n_channels, seq_length)
        # If input is tabular (N, features), we cast it to a univariate sequence (N, 1, features)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
            
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        if self.moment is None:
            self.moment = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs={"task_name": "embedding"}
            )
            self.moment.init()
            self.moment.to(self.device)
            self.moment.eval() # Freeze the foundation model backbone
            
        # 1. Extract Embeddings (Linear Probing setup)
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                out = self.moment(x_enc=batch_X)
                # Average pooling across sequence length -> shape: (batch, d_model)
                emb = out.embeddings
                embeddings_list.append(emb)
                
        embeddings = torch.cat(embeddings_list, dim=0)
        d_model = embeddings.shape[1]
        
        # 2. Initialize and Train the Quantile Head
        self.head = nn.Linear(d_model, 3).to(self.device)
        optimizer = optim.Adam(self.head.parameters(), lr=self.lr)
        
        quantiles = [self.alpha / 2, 0.5, 1.0 - self.alpha / 2]
        dataset = torch.utils.data.TensorDataset(embeddings, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.head.train()
        for epoch in range(self.epochs):
            for batch_emb, batch_y in loader:
                optimizer.zero_grad()
                preds = self.head(batch_emb)
                
                loss_lower = self._pinball_loss(batch_y, preds[:, 0:1], quantiles[0])
                loss_median = self._pinball_loss(batch_y, preds[:, 1:2], quantiles[1])
                loss_upper = self._pinball_loss(batch_y, preds[:, 2:3], quantiles[2])
                
                loss = loss_lower + loss_median + loss_upper
                loss.backward()
                optimizer.step()
                
        return self

    def _get_raw_predictions(self, X):
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
            
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.moment.eval()
        self.head.eval()
        
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                emb = self.moment(x_enc=batch_X).embeddings
                preds = self.head(emb)
                preds_list.append(preds.cpu().numpy())
                
        return np.vstack(preds_list)

    def calibrate(self, X_calib, y_calib):
        """Compute CQR q_score from calibration set."""
        preds = self._get_raw_predictions(X_calib)
        low_calib = preds[:, 0]
        high_calib = preds[:, 2] # Index 2 maps to upper bound
        
        scores = np.maximum(low_calib - y_calib, y_calib - high_calib)
        self.q_score = np.quantile(scores, 1 - self.alpha)

    def get_point_model(self):
        """Return the point-estimate model functionality for analysis utilities."""
        return lambda X: self._get_raw_predictions(X)[:, 1]

    def predict(self, X, quantiles=None, calibrate=True):
        raw_preds = self._get_raw_predictions(X)
        lower = raw_preds[:, 0]
        median = raw_preds[:, 1]
        upper = raw_preds[:, 2]
        
        if quantiles == "mean":
            return median
            
        if isinstance(quantiles, (list, tuple, np.ndarray)):
            if len(quantiles) != 2:
                raise ValueError("quantiles must contain exactly two values: [alpha/2, 1-alpha/2].")
            
            expected = np.array([self.alpha / 2, 1 - self.alpha / 2], dtype=float)
            if not np.allclose(quantiles, expected, atol=1e-8):
                raise ValueError("Only quantiles [alpha/2, 1-alpha/2] are supported for calibrated intervals.")
                
            if calibrate and hasattr(self, 'q_score'):
                lower = lower - self.q_score
                upper = upper + self.q_score
                
                # --- CONSISTENCY CHECKS (Strict Order) ---
                lower = np.maximum(0, lower)
                upper = np.maximum(0, upper)
                median = np.maximum(0, median)

                crossed = lower > upper
                if np.any(crossed):
                    mean_val = (lower[crossed] + upper[crossed]) / 2
                    lower[crossed] = mean_val
                    upper[crossed] = mean_val

                lower = np.minimum(lower, median)
                upper = np.maximum(upper, median)
                upper = np.maximum(upper, lower + 1e-4)
                
            return np.column_stack([lower, upper])
            
        return median