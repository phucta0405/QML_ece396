from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
from sklearn import svm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector


def feature_map(x: np.ndarray) -> QuantumCircuit:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    qc = QuantumCircuit(n)

    qc.h(range(n))

    # -2.0 is to match Havlicek's paper with the definition of RZ and RZZ gates in Qiskit
    for i in range(n):
        qc.rz(-2.0 * x[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            phi_ij = (np.pi - x[i]) * (np.pi - x[j])
            qc.rzz(-2.0 * phi_ij, i, j)

    qc.h(range(n))

    for i in range(n):
        qc.rz(-2.0 * x[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            phi_ij = (np.pi - x[i]) * (np.pi - x[j])
            qc.rzz(-2.0 * phi_ij, i, j)

    return qc


def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same dimension.")
    n = x1.shape[0]

    qc1 = feature_map(x1)
    qc2 = feature_map(x2)

    sv1 = Statevector.from_label("0" * n).evolve(qc1)

    sv2 = Statevector.from_label("0" * n).evolve(qc2)

    inner = np.vdot(sv1.data, sv2.data)

    return float(np.abs(inner) ** 2)

""" This is for shot-based simulation, which is slower in practice.
def kernel_circuit(x1: np.ndarray, x2: np.ndarray, shots=128, backend = DEFAULT_BACKEND) -> float:
    x1 = np.asarray(x1, dtype=float) 
    x2 = np.asarray(x2, dtype=float) 
    if x1.shape != x2.shape: 
        raise ValueError("Input vectors must have the same dimension.") 
    n = x1.shape[0] 

    qc = QuantumCircuit(n) 
    qc.compose(feature_map(x1), inplace=True) 
    qc.compose(feature_map(x2).inverse(), inplace=True) 
    c = ClassicalRegister(n) 
    qc.add_register(c) 
    qc.measure(range(n), range(n)) 

    transpiled_qc = transpile(qc, backend) 
    result = backend.run(transpiled_qc, shots=shots).result() 
    counts = result.get_counts() 
    prob_zero = counts.get('0' * n, 0) / sum(counts.values()) 

    return prob_zero
"""


def compute_quantum_kernel_matrix(X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    
    X1 = np.asarray(X1, dtype=float)
    symmetric = X2 is None
    if symmetric:
        X2 = X1
    else:
        X2 = np.asarray(X2, dtype=float)

    n1, n_features = X1.shape
    n2, n_features_2 = X2.shape
    if n_features != n_features_2:
        raise ValueError("X1 and X2 must have the same number of features.")

    K = np.empty((n1, n2), dtype=float)

    # for speeding up computation in the symmetric case
    if symmetric and n1 == n2:
        # only compute upper triangle and mirror down
        for i in range(n1):
            for j in range(i, n2):
                val = kernel_circuit(X1[i], X2[j])
                K[i, j] = val
                if i != j:
                    K[j, i] = val
    else:
        # otherwise, compute all entries
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_circuit(X1[i], X2[j])

    return K


@dataclass
class QuantumKernelClassifier:
    svc: svm.SVC
    X_train: np.ndarray


def fit_quantumkernel_classifier(X: np.ndarray, Y: np.ndarray, C: float = 1.0) -> QuantumKernelClassifier:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    K_train = compute_quantum_kernel_matrix(X)
    svc_model = svm.SVC(kernel="precomputed", C=C)
    print("Hi")
    svc_model.fit(K_train, Y)
    print("Hi")
    return QuantumKernelClassifier(svc=svc_model, X_train=X)


def predict_quantumkernel_classifier(clf: QuantumKernelClassifier, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    K_test = compute_quantum_kernel_matrix(X, clf.X_train)
    return clf.svc.predict(K_test)

