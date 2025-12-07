import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=4, nhead=2, num_layers=2, dim_feedforward=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)


def fit_transformer_classifier(X: np.ndarray, Y: np.ndarray, num_classes=2, d_model=4, nhead=2, num_layers=2, dim_feedforward=8, dropout=0.1, epochs=100, lr=0.01, verbose=False) -> TransformerClassifier:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)
    input_dim = X.shape[1]
    model = TransformerClassifier(input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
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


def predict_transformer_classifier(model: TransformerClassifier, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds
