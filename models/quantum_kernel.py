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

    # convert data points to statevectors
    dim = 2**n_features
    zero_state = Statevector.from_label("0" * n_features)
    
    # pre-compute statevectors for X1
    states1 = np.zeros((n1, dim), dtype=complex)
    for i in range(n1):
        qc = feature_map(X1[i])
        states1[i] = zero_state.evolve(qc).data

    # pre-compute statevectors for X2
    if symmetric:
        states2 = states1
    else:
        states2 = np.zeros((n2, dim), dtype=complex)
        for i in range(n2):
            qc = feature_map(X2[i])
            states2[i] = zero_state.evolve(qc).data

    # compute the kernel matrix using matrix multiplication
    # K_ij = |<psi_i|psi_j>|^2
    # inner_products matrix shape will be (n1, n2)
    inner_products = states1 @ states2.conj().T
    K = np.abs(inner_products)**2

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
    svc_model.fit(K_train, Y)
    return QuantumKernelClassifier(svc=svc_model, X_train=X)


def predict_quantumkernel_classifier(clf: QuantumKernelClassifier, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    K_test = compute_quantum_kernel_matrix(X, clf.X_train)
    return clf.svc.predict(K_test)