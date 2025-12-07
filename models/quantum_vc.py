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
def vqc_predict_prob(x, params):
    # input number of layers here
    sv = vqc_state(x, params, 2)
    # <Z> = |0|^2 - |1|^2
    p0 = np.abs(sv.data[0]) ** 2
    p1 = np.abs(sv.data[1]) ** 2
    return (p0 - p1 + 1) / 2  # map [-1,1] → [0,1]


# loss function
def loss_fn(params, X, Y):
    preds = np.array([vqc_predict_prob(x, params) for x in X])
    eps = 1e-10
    return -np.mean(Y * np.log(preds + eps) + (1 - Y) * np.log(1 - preds + eps))

# training
@dataclass
class VQCModel:
    params: np.ndarray


def train_vqc(X, Y):
    n_qubits = X.shape[1]
    n_params = 2 * n_qubits   # two-layer ansatz

    # initialize params randomly
    init_params = np.random.uniform(0, 2*np.pi, n_params)

    result = minimize(
        loss_fn, init_params, args=(X, Y),
        method="COBYLA",
        options={"maxiter": 80}
    )

    return VQCModel(params=result.x)

# prediction
def predict_vqc(model, X):
    preds = np.array([vqc_predict_prob(x, model.params) for x in X])
    return (preds > 0.5).astype(int)