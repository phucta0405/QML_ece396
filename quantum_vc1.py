import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from scipy.optimize import minimize
from dataclasses import dataclass

# implementing simple encoding (Stoudenmire and Schwab): xi --> [cos(xi*pi/2), sin(xi*pi/2)]
# try different encodings discussed in Benedetti if time
def encode(x):
    x = np.clip(x, -1.0, 1.0)
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(np.pi * x[i], i)
    return qc

# variational layer (RX, RY, RZ + CZ ring)
def variational_layer(params, n_qubits):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for q in range(n_qubits):
        qc.rx(params[idx], q); idx += 1
        qc.ry(params[idx], q); idx += 1
        qc.rz(params[idx], q); idx += 1
    for q in range(n_qubits):
        qc.cz(q, (q+1)%n_qubits)
    return qc

# prepare state with data re-upload
def prepare_state(x, theta, n_layers):
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)
    qc.compose(encode(x), inplace=True)
    idx = 0
    for _ in range(n_layers):
        slice_params = theta[idx:idx + 3*n_qubits]
        qc.compose(variational_layer(slice_params, n_qubits), inplace=True)
        idx += 3*n_qubits
        qc.compose(encode(x), inplace=True)
    return Statevector.from_label("0"*n_qubits).evolve(qc)

# model output: sum of Z expectations
def model_function(theta, x, n_layers):
    sv = prepare_state(x, theta, n_layers)
    n_qubits = len(x)
    exps = []
    for q in range(n_qubits):
        op = SparsePauliOp.from_list([("I"*q + "Z" + "I"*(n_qubits-q-1), 1.0)])
        exps.append(np.real_if_close(np.real(sv.expectation_value(op))))
    return np.sum(exps)

# loss function
def loss(theta, X, Y, n_layers):
    preds = np.array([model_function(theta, x, n_layers) for x in X])
    probs = (preds + X.shape[1])/(2*X.shape[1])
    eps = 1e-12
    ce = -np.mean(Y*np.log(probs + eps) + (1-Y)*np.log(1-probs + eps))
    return ce

# vqc dataclass
@dataclass
class VQC:
    theta: np.ndarray
    n_layers: int

# train function with progress every 10%
def train_vqc1(X, Y, n_layers=2, maxiter=100, seed=42, batch_size=30):
    rng = np.random.default_rng(seed)
    n_qubits = X.shape[1]
    n_params = n_layers * 3 * n_qubits
    init_theta = rng.uniform(0, 2*np.pi, n_params)

    # SPSA objective function using mini batches
    def objective_function(theta):
        # random batch indices
        idxs = rng.choice(len(X), size=min(len(X), batch_size), replace=False)
        return loss(theta, X[idxs], Y[idxs], n_layers)

    # callback to monitor progress
    loss_history = []
    iteration_count = 0

    def callback(n_fev, params, fval, step_size, accepted):
        nonlocal iteration_count
        loss_history.append(fval)
        iteration_count += 1
        if iteration_count % 10 == 0:
            print(f"Step {iteration_count}/{maxiter} - Batch Loss: {fval:.4f}")

    optimizer = SPSA(maxiter=maxiter, callback=callback, learning_rate=0.1, perturbation=0.1)

    res = optimizer.minimize(fun=objective_function, x0=init_theta)

    return VQC(theta=res.x, n_layers=n_layers)

def predict_vqc1(model: VQC, X):
    preds = np.array([model_function(model.theta, x, model.n_layers) for x in X])
    probs = (preds + X.shape[1])/(2*X.shape[1])
    return (probs > 0.5).astype(int)
