import numpy as np
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

# implementing simple encoding (Stoudenmire and Schwab): xi --> [cos(xi*pi/2), sin(xi*pi/2)]
# try different encodings discussed in Benedetti if time
def encoding_circuit(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.ry(np.pi * x[i], i)

    return qc

# variational circuit
def vc(params, n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)
    idx = 0

    # can change n_layers
    for layer in range(n_layers): 
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1

        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)

    return qc


# full circuit
def vqc_state(x, params, n_layers):
    n_qubits = len(x)

    qc = encoding_circuit(x)
    qc.compose(vc(params, n_qubits, n_layers), inplace=True)

    return Statevector.from_label("0" * n_qubits).evolve(qc)

# prediction: expectation value of Z on qubit 0
def vqc_predict_prob(x, params, n_layers=2):
    sv = vqc_state(x, params, n_layers)
    p0 = np.abs(sv.data[0]) ** 2
    p1 = np.abs(sv.data[1]) ** 2
    return (p0 - p1 + 1) / 2  # map [-1,1] → [0,1]

# batch prediction
def vqc_predict_prob_batch(X, params, n_layers=2):
    # Vectorized batch prediction for X (shape: [num_samples, n_qubits])
    results = []
    for x in X:
        sv = vqc_state(x, params, n_layers)
        p0 = np.abs(sv.data[0]) ** 2
        p1 = np.abs(sv.data[1]) ** 2
        results.append((p0 - p1 + 1) / 2)
    return np.array(results)

# loss function (vectorized)
def loss_fn(params, X, Y, n_layers=2):
    preds = vqc_predict_prob_batch(X, params, n_layers)
    eps = 1e-10
    return -np.mean(Y * np.log(preds + eps) + (1 - Y) * np.log(1 - preds + eps))

# training
@dataclass
class VQCModel:
    params: np.ndarray
    n_layers: int = 2

def train_vqc(X, Y, n_layers=2, maxiter=40):
    n_qubits = X.shape[1]
    n_params = n_layers * n_qubits
    init_params = np.random.uniform(0, 2*np.pi, n_params)
    result = minimize(
        loss_fn, init_params, args=(X, Y, n_layers),
        method="L-BFGS-B",
        options={"maxiter": maxiter}
    )
    return VQCModel(params=result.x, n_layers=n_layers)

# prediction
def predict_vqc(model, X):
    preds = vqc_predict_prob_batch(X, model.params, model.n_layers)
    return (preds > 0.5).astype(int)