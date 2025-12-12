import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
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
def train_vqc1(X, Y, n_layers=2, maxiter=20, seed=42):
    rng = np.random.default_rng(seed)
    n_qubits = X.shape[1]
    n_params = n_layers * 3 * n_qubits
    init_theta = rng.uniform(0, 2*np.pi, n_params)

    iteration_times = []
    progress_percent = np.arange(10, 101, 10)
    next_progress_idx = 0

    def callback(xk):
        nonlocal next_progress_idx
        iteration_times.append(1)
        iter_done = len(iteration_times)
        percent_done = iter_done / maxiter * 100
        if next_progress_idx < len(progress_percent) and percent_done >= progress_percent[next_progress_idx]:
            loss_val = loss(xk, X, Y, n_layers)
            print(f"{progress_percent[next_progress_idx]}% complete - loss: {loss_val:.4f}")
            next_progress_idx += 1

    res = minimize(lambda t: loss(t, X, Y, n_layers),
                   init_theta,
                   method='L-BFGS-B',
                   options={'maxiter': maxiter},
                   callback=callback)
    return VQC(theta=res.x, n_layers=n_layers)

def predict_vqc1(model: VQC, X):
    preds = np.array([model_function(model.theta, x, model.n_layers) for x in X])
    probs = (preds + X.shape[1])/(2*X.shape[1])
    return (probs > 0.5).astype(int)
