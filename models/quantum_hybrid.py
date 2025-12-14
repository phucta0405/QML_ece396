"""
Hybrid Quantum-Classical Neural Network (Autodiff + Adam)
=======================================================

Replaces SPSA with end-to-end differentiation (QNode + PyTorch) and Adam.

Architecture (default):
    X -> Linear -> tanh -> Quantum Layer -> Linear/MLP -> logits
    p = sigmoid(logits)
    CE = -mean(y log p + (1-y) log(1-p)) + L2(theta) + L2(w)

Notes:
- Uses PennyLane QNodes for autodiff + PyTorch integration.
- Tries lightning.gpu (GPU) -> lightning.qubit (fast CPU) -> default.qubit (fallback).
- Supports data re-uploading, optional IsingZZ (RZZ-like) trainable entanglers, and ReLU MLP head.

API matches your project style:
    modelw = fit_hybrid_qnn_classifier(X_train, y_train, ...)
    y_pred = predict_hybrid_qnn_classifier(modelw, X_test)
    p = predict_proba_hybrid_qnn_classifier(modelw, X_test)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml


# -----------------------------------------------------------------------------
# Device selection (GPU if possible for the quantum simulator)
# -----------------------------------------------------------------------------
def make_pl_device(n_qubits: int, prefer_gpu: bool = True) -> Tuple[qml.Device, str]:
    """
    Returns (device, name). Preference order:
      1) lightning.gpu (if installed)
      2) lightning.qubit
      3) default.qubit
    """
    candidates: List[str] = []
    if prefer_gpu:
        candidates.append("lightning.gpu")
    candidates += ["lightning.qubit", "default.qubit"]

    last_err = None
    for name in candidates:
        try:
            dev = qml.device(name, wires=n_qubits, shots=None)  # analytic expvals
            return dev, name
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not create any PennyLane device for n_qubits={n_qubits}. Last error: {last_err}")


# -----------------------------------------------------------------------------
# Quantum circuit (PennyLane) with optional data reupload + optional IsingZZ ring
# -----------------------------------------------------------------------------
def _apply_encoding(x: torch.Tensor, wires: List[int], encoding_scale: float, encoding: str):
    # Keep encoding simple + stable; user can swap in richer maps later.
    # x is assumed already squashed to ~[-1, 1] by tanh in the model.
    if encoding == "angle_ry":
        qml.AngleEmbedding(encoding_scale * np.pi * x, wires=wires, rotation="Y")
    elif encoding == "iqp":
        # Simple IQP-ish: H + RZ + ring ZZ (fixed strength, not trainable here)
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(encoding_scale * np.pi * x[i], wires=w)
        for i in range(len(wires)):
            qml.IsingZZ(0.5 * x[i] * x[(i + 1) % len(wires)], wires=[wires[i], wires[(i + 1) % len(wires)]])
    else:
        # default: "zz_hybrid"-like (RY + fixed ZZ correlations in a ring)
        qml.AngleEmbedding(encoding_scale * np.pi * x, wires=wires, rotation="Y")
        for i in range(len(wires)):
            qml.IsingZZ(0.5 * x[i] * x[(i + 1) % len(wires)], wires=[wires[i], wires[wires[(i + 1) % len(wires)]]])


def build_qnode(
    n_qubits: int,
    n_layers: int,
    encoding: Literal["angle_ry", "zz_hybrid", "iqp"],
    encoding_scale: float,
    entangling: Literal["cnot", "rzz"],
    data_reupload: bool,
    prefer_gpu: bool,
    n_measurements: int = 1,  # 1=Z only, 2=Z+X, 3=Z+X+Y
):
    """
    n_measurements: Number of Pauli observables per qubit (1=Z, 2=Z+X, 3=Z+X+Y)
        More measurements = more quantum features = less need for classical head
    """
    dev, dev_name = make_pl_device(n_qubits, prefer_gpu=prefer_gpu)

    wires = list(range(n_qubits))

    # Choose a diff method likely to work well on the selected backend.
    # - lightning.* usually supports "adjoint" efficiently for expvals
    # - default.qubit supports "backprop"
    diff_method = "adjoint" if dev_name.startswith("lightning") else "backprop"

    # Trainable weights:
    # - rotations: (n_layers, n_qubits, 3) for qml.Rot
    # - entanglers (optional): (n_layers, n_qubits) for ring IsingZZ (RZZ-like)
    # - bias_rotations: (n_qubits, 3) for constant bias-like rotations
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(x, rot_weights, ent_weights=None, bias_rotations=None):
        """
        x: (n_qubits,) - single sample (not batched)
        rot_weights: (n_layers, n_qubits, 3)
        ent_weights: (n_layers, n_qubits) if entangling=="rzz"
        bias_rotations: (n_qubits, 3) for constant bias-like rotations (quantum bias)
        """
        # Apply quantum bias rotations first (constant, data-independent)
        if bias_rotations is not None:
            for q in range(n_qubits):
                a, b, c = bias_rotations[q, 0], bias_rotations[q, 1], bias_rotations[q, 2]
                qml.Rot(a, b, c, wires=wires[q])
        
        # initial encoding
        _apply_encoding(x, wires, encoding_scale=encoding_scale, encoding=encoding)

        for l in range(n_layers):
            # local rotations
            for q in range(n_qubits):
                a, b, c = rot_weights[l, q, 0], rot_weights[l, q, 1], rot_weights[l, q, 2]
                qml.Rot(a, b, c, wires=wires[q])

            # entangling
            if entangling == "rzz":
                # ring IsingZZ with trainable angles
                for q in range(n_qubits):
                    q2 = (q + 1) % n_qubits
                    qml.IsingZZ(ent_weights[l, q], wires=[wires[q], wires[q2]])
            else:
                # fixed CNOT ring
                for q in range(n_qubits):
                    q2 = (q + 1) % n_qubits
                    qml.CNOT(wires=[wires[q], wires[q2]])

            # data re-upload (between layers)
            if data_reupload and (l < n_layers - 1):
                _apply_encoding(x, wires, encoding_scale=encoding_scale, encoding=encoding)

        # Measure multiple observables per qubit for richer quantum features
        measurements = []
        for i in wires:
            measurements.append(qml.expval(qml.PauliZ(i)))  # Always measure Z
            if n_measurements >= 2:
                measurements.append(qml.expval(qml.PauliX(i)))  # Also X
            if n_measurements >= 3:
                measurements.append(qml.expval(qml.PauliY(i)))  # Also Y
        return measurements  # shape: (n_qubits * n_measurements,)

    return qnode, dev_name


# -----------------------------------------------------------------------------
# Torch Module: Quantum Layer (wraps the QNode)
# -----------------------------------------------------------------------------
class QuantumLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        encoding: Literal["angle_ry", "zz_hybrid", "iqp"] = "zz_hybrid",
        encoding_scale: float = 2.0,
        entangling: Literal["cnot", "rzz"] = "rzz",
        data_reupload: bool = True,
        prefer_gpu: bool = True,
        n_measurements: int = 1,  # 1=Z only, 2=Z+X, 3=Z+X+Y
        use_quantum_bias: bool = True,  # Use quantum rotations as bias
        use_quantum_weights: bool = True,  # Use quantum scaling as weights
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding
        self.encoding_scale = float(encoding_scale)
        self.entangling = entangling
        self.data_reupload = bool(data_reupload)
        self.n_measurements = int(n_measurements)
        self.output_dim = self.n_qubits * self.n_measurements  # Output dimension
        self.use_quantum_bias = bool(use_quantum_bias)
        self.use_quantum_weights = bool(use_quantum_weights)

        self.qnode, self.backend_name = build_qnode(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            encoding=self.encoding,
            encoding_scale=self.encoding_scale,
            entangling=self.entangling,
            data_reupload=self.data_reupload,
            prefer_gpu=prefer_gpu,
            n_measurements=self.n_measurements,
        )

        # Trainable params (torch)
        self.rot = nn.Parameter(0.01 * torch.randn(self.n_layers, self.n_qubits, 3))
        if self.entangling == "rzz":
            self.ent = nn.Parameter(torch.zeros(self.n_layers, self.n_qubits))
        else:
            self.ent = None
        
        # Quantum bias: constant rotations (like classical bias)
        if self.use_quantum_bias:
            self.quantum_bias = nn.Parameter(0.01 * torch.randn(self.n_qubits, 3))
        else:
            self.quantum_bias = None
        
        # Quantum weights: scaling factors for measurements (like classical weights)
        if self.use_quantum_weights:
            self.quantum_weight = nn.Parameter(torch.ones(self.output_dim))  # Initialize to 1.0
        else:
            self.quantum_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_qubits) float tensor (typically already in [-1,1] via tanh)
        returns: (B, n_qubits * n_measurements) with quantum measurements
        """
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"QuantumLayer expects x shape (B, {self.n_qubits}), got {tuple(x.shape)}")

        batch_size = x.shape[0]
        # Process each sample in the batch individually
        # Detach input data - the input data itself shouldn't be trainable, only quantum parameters
        x_detached = x.detach().requires_grad_(False)
        
        results = []
        for i in range(batch_size):
            x_sample = x_detached[i]  # (n_qubits,)
            # Pass quantum bias to qnode
            bias_rot = self.quantum_bias if self.use_quantum_bias else None
            if self.entangling == "rzz":
                z_sample = self.qnode(x_sample, self.rot, self.ent, bias_rot)
            else:
                z_sample = self.qnode(x_sample, self.rot, None, bias_rot)
            # z_sample is a list of expvals, convert to tensor
            if isinstance(z_sample, list):
                z_sample = torch.stack(z_sample)  # (n_qubits * n_measurements,)
            
            # Apply quantum weights (scaling like classical weights)
            if self.use_quantum_weights:
                z_sample = z_sample * self.quantum_weight
            
            results.append(z_sample)
        
        # Stack results: (B, n_qubits * n_measurements)
        z = torch.stack(results, dim=0)
        return z


# -----------------------------------------------------------------------------
# Full Hybrid Model (classical -> quantum -> classical)
# -----------------------------------------------------------------------------
class HybridQNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        encoding: Literal["angle_ry", "zz_hybrid", "iqp"] = "zz_hybrid",
        encoding_scale: float = 2.0,
        entangling: Literal["cnot", "rzz"] = "rzz",
        data_reupload: bool = True,
        prefer_q_gpu: bool = True,
        head: Literal["logistic", "mlp", "minimal", "quantum_only"] = "mlp",
        hidden_dim: int = 8,
        n_measurements: int = 1,  # 1=Z only, 2=Z+X, 3=Z+X+Y (more = more quantum)
        use_quantum_bias: bool = True,  # Use quantum rotations as bias
        use_quantum_weights: bool = True,  # Use quantum scaling as weights
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)

        # Classical pre-net: map -> n_qubits, then squash for stable rotations
        self.pre = nn.Sequential(
            nn.Linear(self.input_dim, self.n_qubits),
            nn.Tanh(),
        )

        self.quantum = QuantumLayer(
            n_qubits=self.n_qubits,
            n_layers=n_q_layers,
            encoding=encoding,
            encoding_scale=encoding_scale,
            entangling=entangling,
            data_reupload=data_reupload,
            prefer_gpu=prefer_q_gpu,
            n_measurements=n_measurements,
            use_quantum_bias=use_quantum_bias,
            use_quantum_weights=use_quantum_weights,
        )

        # Classical post-net: 
        # - "quantum_only": no head, just sum (pure quantum)
        # - "minimal": single linear layer with NO bias (quantum bias replaces it)
        # - "logistic": single linear layer with bias (kept for compatibility)
        # - "mlp": small ReLU MLP (more classical)
        quantum_output_dim = self.quantum.output_dim  # n_qubits * n_measurements
        if head == "quantum_only":
            # Pure quantum: just sum the quantum features (no classical parameters)
            self.head = None
        elif head == "minimal":
            # Minimal: single weight vector, NO bias (quantum bias/weights handle it)
            self.head = nn.Linear(quantum_output_dim, 1, bias=False)
        elif head == "logistic":
            self.head = nn.Linear(quantum_output_dim, 1)  # With bias
        else:
            # Reduce hidden_dim proportionally if we have more quantum features
            effective_hidden = max(4, hidden_dim // max(1, n_measurements - 1)) if n_measurements > 1 else hidden_dim
            # Use bias=False in first layer if quantum bias is enabled
            first_bias = not use_quantum_bias
            self.head = nn.Sequential(
                nn.Linear(quantum_output_dim, effective_hidden, bias=first_bias),
                nn.ReLU(),
                nn.Linear(effective_hidden, 1, bias=False),  # No bias in final layer
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns logits (shape: (B,))
        """
        x = self.pre(x)                 # (B, n_qubits)
        z = self.quantum(x)             # (B, n_qubits * n_measurements)
        if self.head is None:
            # Pure quantum: sum all quantum features
            logits = z.sum(dim=1)  # (B,)
        else:
            logits = self.head(z).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# Wrapper + training API
# -----------------------------------------------------------------------------
@dataclass
class HybridQNNModel:
    model: HybridQNN
    input_dim: int
    backend_name: str
    history: List[float]


def fit_hybrid_qnn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: Optional[int] = None,
    n_q_layers: int = 2,
    encoding: Literal["angle_ry", "zz_hybrid", "iqp"] = "zz_hybrid",
    encoding_scale: float = 2.0,
    entangling: Literal["cnot", "rzz"] = "rzz",
    data_reupload: bool = True,
    head: Literal["logistic", "mlp", "minimal", "quantum_only"] = "quantum_only",
    hidden_dim: int = 8,
    n_measurements: int = 1,  # 1=Z only, 2=Z+X, 3=Z+X+Y (more = more quantum features)
    use_quantum_bias: bool = True,  # Use quantum rotations as bias (replaces classical bias)
    use_quantum_weights: bool = True,  # Use quantum scaling as weights (replaces classical weights)
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-2,
    lambda_theta: float = 1e-4,   # L2 on quantum params
    lambda_w: float = 1e-4,       # L2 on head weights
    grad_clip: float = 1.0,
    prefer_q_gpu: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> HybridQNNModel:
    """
    Binary classifier training with Adam + BCEWithLogitsLoss + explicit L2 regs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

    n_samples, n_features = X_train.shape
    if n_qubits is None:
        n_qubits = min(n_features, 4)

    model = HybridQNN(
        input_dim=n_features,
        n_qubits=n_qubits,
        n_q_layers=n_q_layers,
        encoding=encoding,
        encoding_scale=encoding_scale,
        entangling=entangling,
        data_reupload=data_reupload,
        prefer_q_gpu=prefer_q_gpu,
        head=head,
        hidden_dim=hidden_dim,
        n_measurements=n_measurements,
        use_quantum_bias=use_quantum_bias,
        use_quantum_weights=use_quantum_weights,
    )

    # Keep the torch model on CPU (safe with PL devices). The quantum simulator can still be GPU-accelerated.
    device = torch.device("cpu")
    model.to(device)

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    history: List[float] = []

    for ep in range(epochs):
        model.train()
        running = 0.0
        nb = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            ce = criterion(logits, yb)

            # L2(theta): all quantum params
            l2_theta = 0.0
            for p in model.quantum.parameters():
                l2_theta = l2_theta + torch.sum(p * p)

            # L2(w): all linear weights in head (bias excluded)
            l2_w_term = 0.0
            if model.head is not None:
                for m in model.head.modules():
                    if isinstance(m, nn.Linear):
                        l2_w_term = l2_w_term + torch.sum(m.weight * m.weight)

            loss = ce + lambda_theta * l2_theta + lambda_w * l2_w_term

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            opt.step()

            running += float(loss.item())
            nb += 1

        avg = running / max(1, nb)
        history.append(avg)

        # Compute accuracy on training set after each epoch
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_train).to(device)
            y_tensor = torch.from_numpy(y_train).to(device)
            logits = model(X_tensor)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_tensor).float().mean().item()

        # Print loss and accuracy per epoch
        print(f"Epoch {ep+1}/{epochs}  loss={avg:.4f}  acc={acc:.4f}")

    return HybridQNNModel(
        model=model,
        input_dim=n_features,
        backend_name=model.quantum.backend_name,
        history=history,
    )


def predict_proba_hybrid_qnn_classifier(modelw: HybridQNNModel, X: np.ndarray) -> np.ndarray:
    model = modelw.model
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        p = torch.sigmoid(logits)
    return p.cpu().numpy()


def predict_hybrid_qnn_classifier(modelw: HybridQNNModel, X: np.ndarray) -> np.ndarray:
    p = predict_proba_hybrid_qnn_classifier(modelw, X)
    return (p > 0.5).astype(int)
