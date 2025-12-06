import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, num_layers=2, residual=False, num_classes=2, dropout_rate: Optional[float] = None):
        super().__init__()
        self.residual = residual
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, num_classes) # Output for classification
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            prev = out
            out = self.activation(layer(out))
            if self.dropout is not None:
                out = self.dropout(out)
            if self.residual:
                out = out + prev
        return self.output_layer(out)


def fit_feedforward_classifier(X: np.ndarray, Y: np.ndarray, hidden_dim=8, num_layers=2, residual=False, num_classes=2, epochs=100, lr=0.01, verbose=False, dropout_rate: Optional[float] = None)  -> FeedForwardNet:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)
    input_dim = X.shape[1]
    model = FeedForwardNet(input_dim, hidden_dim, num_layers, residual, num_classes, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model


def predict_feedforward_classifier(model: FeedForwardNet, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        # selects the index of the largest value in the output logits for each input sample
        preds = torch.argmax(logits, dim=1).cpu().numpy() 
    return preds
