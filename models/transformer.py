import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=1, nhead=2,
                 num_layers=1, dim_feedforward=2, dropout=0, use_cls=True):
        super().__init__()
        self.input_dim = input_dim
        self.use_cls = use_cls

        self.token_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim + (1 if use_cls else 0), d_model))

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.token_proj(x) 

        if self.use_cls:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1) 
            x = torch.cat([cls, x], dim=1)         

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)

        # pool
        if self.use_cls:
            h = x[:, 0, :]           # CLS token
        else:
            h = x.mean(dim=1)        # mean pool over feature tokens

        return self.classifier(h)



def fit_transformer_classifier(X: np.ndarray, Y: np.ndarray, num_classes=2, d_model=2, nhead=2, num_layers=1, dim_feedforward=8, dropout=0.1, epochs=100, lr=0.001, verbose=False, use_cls=True) -> TransformerClassifier:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)
    input_dim = X.shape[1]
    model = TransformerClassifier(input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout, use_cls).to(device)
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
