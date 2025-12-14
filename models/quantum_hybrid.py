"""
Hybrid Quantum-Classical Neural Network - IMPROVED ENCODING
============================================================

Based on 2024-2025 QML research, this version adds:
1. ZZ feature correlations in encoding (captures x_i * x_j interactions)
2. Learnable input scaling (optimizes Fourier frequencies per feature)
3. Multiple encoding strategies (ANGLE_RY, ZZ_HYBRID, IQP)

Usage:
    model = fit_hybrid_qnn_classifier(X_train, y_train, encoding='zz_hybrid')
    predictions = predict_hybrid_qnn_classifier(model, X_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


# =============================================================================
# Backend Setup
# =============================================================================

def _make_aer_backend(prefer_gpu: bool = True):
    """Create Aer backend with GPU if available."""
    if prefer_gpu:
        try:
            sim = AerSimulator(method="statevector", device="GPU")
            test_qc = QuantumCircuit(1)
            test_qc.h(0)
            test_qc.save_statevector()
            try:
                job = sim.run(test_qc)
                job.result()
                return sim, True
            except RuntimeError:
                pass
        except Exception:
            pass
    
    sim = AerSimulator(method="statevector", device="CPU")
    return sim, False


def _precompute_z_signs(n_qubits: int) -> np.ndarray:
    """Precompute sign matrix for Z expectations."""
    n_states = 2**n_qubits
    S = np.zeros((n_qubits, n_states), dtype=np.float64)
    
    for i in range(n_qubits):
        for basis_idx in range(n_states):
            if (basis_idx >> (n_qubits - 1 - i)) & 1:
                S[i, basis_idx] = -1.0
            else:
                S[i, basis_idx] = 1.0
    
    return S


# =============================================================================
# Quantum Layer with Improved Encoding
# =============================================================================

class QuantumLayerImproved(nn.Module):
    """
    Quantum layer with research-backed encoding improvements.
    
    Encoding options:
    - 'angle_ry': Basic RY encoding (baseline)
    - 'zz_hybrid': RY + ZZ correlations (recommended, captures feature interactions)
    - 'iqp': Hadamard + RZ + ZZ (highest expressibility)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        prefer_aer_gpu: bool = True,
        data_reupload: bool = True,
        encoding: Literal['angle_ry', 'zz_hybrid', 'iqp'] = 'zz_hybrid',
        encoding_scale: float = 2.0,
        learnable_scaling: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.data_reupload = data_reupload
        self.encoding = encoding
        self.encoding_scale = encoding_scale
        self.learnable_scaling = learnable_scaling

        # Variational parameters
        self.n_theta = n_layers * 3 * n_qubits
        self.n_gamma = n_layers * n_qubits
        
        # Xavier-like initialization
        theta_scale = np.sqrt(2.0 / (3 * n_qubits))
        self.theta = nn.Parameter(theta_scale * torch.randn(self.n_theta))
        self.gamma = nn.Parameter(0.1 * torch.randn(self.n_gamma))
        
        # Learnable input scaling (research finding: 10-41% accuracy difference)
        if learnable_scaling:
            self.input_scale = nn.Parameter(torch.ones(n_qubits))
        else:
            self.register_buffer('input_scale', torch.ones(n_qubits))
        
        # ZZ encoding strength (for zz_hybrid and iqp)
        if encoding in ['zz_hybrid', 'iqp']:
            n_pairs = n_qubits * (n_qubits - 1) // 2
            self.zz_scale = nn.Parameter(0.5 * torch.ones(max(n_pairs, 1)))
        else:
            self.register_buffer('zz_scale', torch.zeros(1))

        self.backend, self.using_gpu = _make_aer_backend(prefer_gpu=prefer_aer_gpu)
        self._S = _precompute_z_signs(n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        """Build parameterized quantum circuit."""
        self.xp = ParameterVector("x", self.n_qubits)
        self.sp = ParameterVector("s", self.n_qubits)  # input scaling
        self.tp = ParameterVector("t", self.n_theta)
        self.gp = ParameterVector("g", self.n_gamma)
        
        # ZZ parameters for encoding
        n_pairs = self.n_qubits * (self.n_qubits - 1) // 2
        self.zzp = ParameterVector("zz", max(n_pairs, 1))

        qc = QuantumCircuit(self.n_qubits)

        # Initial data encoding
        self._add_encoding(qc)

        # Variational layers with optional data re-uploading
        idx_t = 0
        idx_g = 0
        for layer in range(self.n_layers):
            # Variational rotations
            for i in range(self.n_qubits):
                qc.rx(self.tp[idx_t], i); idx_t += 1
                qc.ry(self.tp[idx_t], i); idx_t += 1
                qc.rz(self.tp[idx_t], i); idx_t += 1

            # Entangling layer
            for i in range(self.n_qubits):
                qc.rzz(self.gp[idx_g], i, (i + 1) % self.n_qubits)
                idx_g += 1

            # Data re-uploading (proven to increase expressibility)
            if self.data_reupload and layer < self.n_layers - 1:
                self._add_encoding(qc)

        qc.save_statevector()
        self.qc = qc

    def _add_encoding(self, qc: QuantumCircuit):
        """Add data encoding block based on encoding strategy."""
        
        if self.encoding == 'angle_ry':
            # Basic RY encoding
            for i in range(self.n_qubits):
                angle = self.encoding_scale * np.pi * self.sp[i] * self.xp[i]
                qc.ry(angle, i)
                
        elif self.encoding == 'zz_hybrid':
            # RY encoding + ZZ correlations (recommended)
            for i in range(self.n_qubits):
                angle = self.encoding_scale * np.pi * self.sp[i] * self.xp[i]
                qc.ry(angle, i)
            
            # ZZ gates capture x_i * x_j feature correlations
            pair_idx = 0
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.rzz(self.zzp[pair_idx] * self.xp[i] * self.xp[j], i, j)
                    pair_idx += 1
                    
        elif self.encoding == 'iqp':
            # IQP-style: H + RZ + ZZ (highest expressibility)
            for i in range(self.n_qubits):
                qc.h(i)
            for i in range(self.n_qubits):
                angle = self.encoding_scale * np.pi * self.sp[i] * self.xp[i]
                qc.rz(angle, i)
            
            pair_idx = 0
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.rzz(self.zzp[pair_idx] * self.xp[i] * self.xp[j], i, j)
                    pair_idx += 1

    def _batch_run_circuits(self, x_batch, theta_vals, gamma_vals, 
                            input_scale_vals, zz_scale_vals):
        """Run multiple circuit evaluations in one Aer job."""
        N = x_batch.shape[0]
        circuits = []
        n_pairs = self.n_qubits * (self.n_qubits - 1) // 2
        
        for i in range(N):
            param_dict = {}
            
            for j in range(self.n_qubits):
                x_val = np.tanh(x_batch[i, j])
                param_dict[self.xp[j]] = float(x_val)
                param_dict[self.sp[j]] = float(input_scale_vals[j])
            
            for j in range(self.n_theta):
                param_dict[self.tp[j]] = float(theta_vals[i, j])
            for j in range(self.n_gamma):
                param_dict[self.gp[j]] = float(gamma_vals[i, j])
            
            for j in range(max(n_pairs, 1)):
                if j < len(zz_scale_vals):
                    param_dict[self.zzp[j]] = float(zz_scale_vals[j])
                else:
                    param_dict[self.zzp[j]] = 0.0
            
            circuits.append(self.qc.assign_parameters(param_dict))
        
        job = self.backend.run(circuits)
        result = job.result()
        
        outputs = np.zeros((N, self.n_qubits), dtype=np.float64)
        for i in range(N):
            sv = result.get_statevector(i).data
            probs = np.abs(sv) ** 2
            outputs[i] = self._S @ probs
        
        return outputs

    def _batch_forward(self, x_batch, theta_np, gamma_np, input_scale_np, zz_scale_np):
        """Forward pass for batch."""
        B = x_batch.shape[0]
        theta_batch = np.tile(theta_np, (B, 1))
        gamma_batch = np.tile(gamma_np, (B, 1))
        return self._batch_run_circuits(x_batch, theta_batch, gamma_batch, 
                                        input_scale_np, zz_scale_np)

    def _compute_parameter_shift_gradients(self, x_batch, theta_np, gamma_np,
                                            input_scale_np, zz_scale_np, grad_output):
        """Compute gradients using parameter-shift rule."""
        shift = np.pi / 2.0
        
        grad_theta = np.zeros(self.n_theta, dtype=np.float64)
        grad_gamma = np.zeros(self.n_gamma, dtype=np.float64)
        grad_input_scale = np.zeros(self.n_qubits, dtype=np.float64)
        grad_zz_scale = np.zeros_like(zz_scale_np)
        
        # Theta gradients
        for param_idx in range(self.n_theta):
            theta_plus = theta_np.copy()
            theta_plus[param_idx] += shift
            theta_minus = theta_np.copy()
            theta_minus[param_idx] -= shift
            
            out_plus = self._batch_forward(x_batch, theta_plus, gamma_np, 
                                           input_scale_np, zz_scale_np)
            out_minus = self._batch_forward(x_batch, theta_minus, gamma_np,
                                            input_scale_np, zz_scale_np)
            
            d_out = 0.5 * (out_plus - out_minus)
            grad_theta[param_idx] = np.sum(grad_output * d_out)
        
        # Gamma gradients
        for param_idx in range(self.n_gamma):
            gamma_plus = gamma_np.copy()
            gamma_plus[param_idx] += shift
            gamma_minus = gamma_np.copy()
            gamma_minus[param_idx] -= shift
            
            out_plus = self._batch_forward(x_batch, theta_np, gamma_plus,
                                           input_scale_np, zz_scale_np)
            out_minus = self._batch_forward(x_batch, theta_np, gamma_minus,
                                            input_scale_np, zz_scale_np)
            
            d_out = 0.5 * (out_plus - out_minus)
            grad_gamma[param_idx] = np.sum(grad_output * d_out)
        
        # Input scale gradients (if learnable)
        if self.learnable_scaling:
            for param_idx in range(self.n_qubits):
                scale_plus = input_scale_np.copy()
                scale_plus[param_idx] += shift
                scale_minus = input_scale_np.copy()
                scale_minus[param_idx] -= shift
                
                out_plus = self._batch_forward(x_batch, theta_np, gamma_np,
                                               scale_plus, zz_scale_np)
                out_minus = self._batch_forward(x_batch, theta_np, gamma_np,
                                                scale_minus, zz_scale_np)
                
                d_out = 0.5 * (out_plus - out_minus)
                grad_input_scale[param_idx] = np.sum(grad_output * d_out)
        
        # ZZ scale gradients
        if self.encoding in ['zz_hybrid', 'iqp']:
            for param_idx in range(len(zz_scale_np)):
                zz_plus = zz_scale_np.copy()
                zz_plus[param_idx] += shift
                zz_minus = zz_scale_np.copy()
                zz_minus[param_idx] -= shift
                
                out_plus = self._batch_forward(x_batch, theta_np, gamma_np,
                                               input_scale_np, zz_plus)
                out_minus = self._batch_forward(x_batch, theta_np, gamma_np,
                                                input_scale_np, zz_minus)
                
                d_out = 0.5 * (out_plus - out_minus)
                grad_zz_scale[param_idx] = np.sum(grad_output * d_out)
        
        return grad_theta, grad_gamma, grad_input_scale, grad_zz_scale

    def forward(self, x):
        """Forward pass."""
        if x.shape[1] < self.n_qubits:
            pad = self.n_qubits - x.shape[1]
            x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)], dim=1)
        elif x.shape[1] > self.n_qubits:
            x = x[:, :self.n_qubits]
        
        x = torch.tanh(x)
        return QuantumFunctionImproved.apply(x, self.theta, self.gamma, 
                                              self.input_scale, self.zz_scale, self)


class QuantumFunctionImproved(torch.autograd.Function):
    """Custom autograd for improved quantum layer."""
    
    @staticmethod
    def forward(ctx, x, theta, gamma, input_scale, zz_scale, layer):
        ctx.layer = layer
        
        x_np = x.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        gamma_np = gamma.detach().cpu().numpy()
        input_scale_np = input_scale.detach().cpu().numpy()
        zz_scale_np = zz_scale.detach().cpu().numpy()
        
        output_np = layer._batch_forward(x_np, theta_np, gamma_np, 
                                         input_scale_np, zz_scale_np)
        output = torch.tensor(output_np, dtype=x.dtype, device=x.device)
        
        ctx.save_for_backward(x, theta, gamma, input_scale, zz_scale)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        layer = ctx.layer
        x, theta, gamma, input_scale, zz_scale = ctx.saved_tensors
        
        x_np = x.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        gamma_np = gamma.detach().cpu().numpy()
        input_scale_np = input_scale.detach().cpu().numpy()
        zz_scale_np = zz_scale.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_theta_np, grad_gamma_np, grad_scale_np, grad_zz_np = \
            layer._compute_parameter_shift_gradients(
                x_np, theta_np, gamma_np, input_scale_np, zz_scale_np, grad_output_np
            )
        
        grad_theta = torch.tensor(grad_theta_np, dtype=theta.dtype, device=theta.device)
        grad_gamma = torch.tensor(grad_gamma_np, dtype=gamma.dtype, device=gamma.device)
        grad_input_scale = torch.tensor(grad_scale_np, dtype=input_scale.dtype, device=input_scale.device)
        grad_zz_scale = torch.tensor(grad_zz_np, dtype=zz_scale.dtype, device=zz_scale.device)
        
        return None, grad_theta, grad_gamma, grad_input_scale, grad_zz_scale, None


# =============================================================================
# Hybrid Model
# =============================================================================

class HybridQNN(nn.Module):
    """Hybrid quantum-classical neural network."""
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        hidden_dim: int = 8,
        n_quantum_layers: int = 2,
        output_dim: int = 2,
        prefer_aer_gpu: bool = True,
        encoding: str = 'zz_hybrid',
        data_reupload: bool = True,
        encoding_scale: float = 2.0,
        learnable_scaling: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_qubits)
        
        self.quantum_layer = QuantumLayerImproved(
            n_qubits=n_qubits,
            n_layers=n_quantum_layers,
            prefer_aer_gpu=prefer_aer_gpu,
            data_reupload=data_reupload,
            encoding=encoding,
            encoding_scale=encoding_scale,
            learnable_scaling=learnable_scaling,
        )
        
        self.fc2 = nn.Linear(n_qubits, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum_layer(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# =============================================================================
# Model Wrapper
# =============================================================================

@dataclass
class HybridQNNModel:
    """Wrapper for trained model."""
    model: HybridQNN
    input_dim: int
    device: torch.device
    using_aer_gpu: bool
    encoding: str


# =============================================================================
# Training & Prediction Functions
# =============================================================================

def fit_hybrid_qnn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int = None,
    hidden_dim: int = 16,
    n_quantum_layers: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 8,
    verbose: bool = True,
    prefer_torch_cuda: bool = True,
    prefer_aer_gpu: bool = True,
    encoding: Literal['angle_ry', 'zz_hybrid', 'iqp'] = 'zz_hybrid',
    data_reupload: bool = True,
    encoding_scale: float = 2.0,
    learnable_scaling: bool = True,
    l1_reg: float = 0.1,
) -> HybridQNNModel:
    """
    Train a hybrid quantum-classical neural network classifier.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_samples, n_features)
    y_train : np.ndarray
        Training labels, shape (n_samples,)
    n_qubits : int, optional
        Number of qubits. Default: min(n_features, 4)
    hidden_dim : int
        Hidden layer dimension
    n_quantum_layers : int
        Number of variational quantum layers
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    verbose : bool
        Print training progress
    encoding : str
        Encoding strategy:
        - 'angle_ry': Basic RY encoding (baseline)
        - 'zz_hybrid': RY + ZZ correlations (recommended)
        - 'iqp': H + RZ + ZZ (highest expressibility)
    data_reupload : bool
        Re-encode data before each layer (increases expressibility)
    encoding_scale : float
        Scale factor for encoding angles (2.0 = 2π range)
    learnable_scaling : bool
        Learn per-feature input scaling
    l1_reg : float
        L1 regularization strength
        
    Returns
    -------
    HybridQNNModel
        Trained model wrapper
    """
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    if n_qubits is None:
        n_qubits = min(n_features, 4)

    device = torch.device("cuda:0" if (prefer_torch_cuda and torch.cuda.is_available()) else "cpu")

    model = HybridQNN(
        input_dim=n_features,
        n_qubits=n_qubits,
        hidden_dim=hidden_dim,
        n_quantum_layers=n_quantum_layers,
        output_dim=n_classes,
        prefer_aer_gpu=prefer_aer_gpu,
        encoding=encoding,
        data_reupload=data_reupload,
        encoding_scale=encoding_scale,
        learnable_scaling=learnable_scaling,
    ).to(device)

    if verbose:
        print("Training Hybrid QNN (Improved Encoding):")
        print(f"  Device: {device}, Aer GPU: {model.quantum_layer.using_gpu}")
        print(f"  Input: {n_features}, Qubits: {n_qubits}, Layers: {n_quantum_layers}")
        print(f"  Encoding: {encoding}, Data re-upload: {data_reupload}")
        print(f"  Learnable scaling: {learnable_scaling}")
        print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_full = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_full = torch.tensor(y_train, dtype=torch.long, device=device)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            if l1_reg > 0:
                loss = loss + l1_reg * torch.norm(model.quantum_layer.theta, p=1)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs_full = model(X_full)
                pred = torch.argmax(outputs_full, dim=1)
                acc = (pred == y_full).float().mean().item()
            model.train()
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}]  loss={avg_loss:.4f}  acc={acc*100:.2f}%")

    if verbose:
        print("Training complete!")

    return HybridQNNModel(
        model=model,
        input_dim=n_features,
        device=device,
        using_aer_gpu=model.quantum_layer.using_gpu,
        encoding=encoding,
    )


def predict_hybrid_qnn_classifier(
    model_wrapper: HybridQNNModel, 
    X_test: np.ndarray
) -> np.ndarray:
    """
    Make predictions with trained hybrid QNN.
    
    Parameters
    ----------
    model_wrapper : HybridQNNModel
        Trained model from fit_hybrid_qnn_classifier
    X_test : np.ndarray
        Test features, shape (n_samples, n_features)
        
    Returns
    -------
    np.ndarray
        Predicted class labels
    """
    model = model_wrapper.model
    device = model_wrapper.device
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted = torch.argmax(outputs, dim=1)
    return predicted.cpu().numpy()



def get_quantum_features(
    model_wrapper: HybridQNNModel, 
    X: np.ndarray
) -> np.ndarray:
    """
    Extract quantum layer features.
    
    Parameters
    ----------
    model_wrapper : HybridQNNModel
        Trained model
    X : np.ndarray
        Input features
        
    Returns
    -------
    np.ndarray
        Quantum layer outputs, shape (n_samples, n_qubits)
    """
    model = model_wrapper.model
    device = model_wrapper.device
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        x = torch.tanh(model.fc1(X_tensor))
        q_feats = model.quantum_layer(x)
    return q_feats.cpu().numpy()


def print_model_info(model_wrapper: HybridQNNModel):
    """Print model configuration and parameter counts."""
    model = model_wrapper.model
    qlayer = model.quantum_layer
    
    print("\n" + "="*60)
    print("HYBRID QNN MODEL INFO")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Encoding strategy: {model_wrapper.encoding}")
    print(f"Using Aer GPU: {model_wrapper.using_aer_gpu}")
    
    print(f"\nQuantum layer:")
    print(f"  Qubits: {qlayer.n_qubits}")
    print(f"  Layers: {qlayer.n_layers}")
    print(f"  Data re-upload: {qlayer.data_reupload}")
    print(f"  Learnable scaling: {qlayer.learnable_scaling}")
    
    if qlayer.learnable_scaling:
        scale = qlayer.input_scale.detach().cpu().numpy()
        print(f"  Learned input scales: {scale}")
    
    if qlayer.encoding in ['zz_hybrid', 'iqp']:
        zz = qlayer.zz_scale.detach().cpu().numpy()
        print(f"  Learned ZZ scales: {zz}")
    
    print("="*60)


# =============================================================================
# Main - Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HYBRID QNN WITH IMPROVED ENCODING")
    print("="*70)

    from sklearn.datasets import make_moons, make_circles
    from sklearn.model_selection import train_test_split
    import time

    # Test on circles (requires feature correlations)
    print("\n[Dataset: make_circles]")
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train with zz_hybrid encoding (recommended)
    print(f"\n{'='*70}")
    print("ENCODING: zz_hybrid (recommended)")
    print('='*70)
    
    start = time.time()
    model = fit_hybrid_qnn_classifier(
        X_train, y_train,
        n_qubits=2,
        hidden_dim=4,
        n_quantum_layers=2,
        epochs=50,
        lr=0.01,
        batch_size=8,
        verbose=True,
        encoding='zz_hybrid',
        learnable_scaling=True,
    )
    elapsed = time.time() - start
    
    y_pred = predict_hybrid_qnn_classifier(model, X_test)
    acc = np.mean(y_pred == y_test)
    
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Training time: {elapsed:.1f}s")
    
    print_model_info(model)