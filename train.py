import numpy as np
import argparse
import random
import torch
from sklearn.model_selection import train_test_split
from basic_datasets import get_bas_example, nsphere_sample, Spiral_sample, Spiral_sample2, Noisy_nsphere_sample, get_noisy_bas_example
from models.classical_svm import fit_svm_classifier, predict_svm_classifier
from models.feed_forward_net import fit_feedforward_classifier, predict_feedforward_classifier
from models.transformer import fit_transformer_classifier, predict_transformer_classifier
from models.quantum_kernel import fit_quantumkernel_classifier, predict_quantumkernel_classifier
from quantum_vc1 import train_vqc1, predict_vqc1

number_of_samples = 1000

"""
# Set global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass
"""


DATASET_LOADERS = {
    'bas': get_bas_example,
    'nsphere': lambda: nsphere_sample(number_of_samples, ndim=2),
    'circles': lambda: None,  # sphere but 2 dimensions
    'spiral': lambda: Spiral_sample(dW=0.1, Ns=number_of_samples),
    'spiral2': lambda: Spiral_sample2(dW=0.1, Ns=number_of_samples),
}

MODELS = {
    'classical_svm': (fit_svm_classifier, predict_svm_classifier),
    "feedforward": (fit_feedforward_classifier, predict_feedforward_classifier),
    'transformer': (fit_transformer_classifier, predict_transformer_classifier),
    'quantum_kernel': (fit_quantumkernel_classifier, predict_quantumkernel_classifier),
    'quantum_vc1': (train_vqc1, predict_vqc1),
}

def train_and_test(dataset: str, model: str):
    if model not in MODELS:
        print(f"Model '{model}' not implemented yet. Use existing models.")
        return
    fit_fn, predict_fn = MODELS[model]
    n_runs = 5
    accs = []
    for run in range(n_runs):
        if dataset == 'bas':
            x_BAS, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary = get_bas_example()
            X_train, X_test, y_train, y_test = train_test_split(x_BAS, Y_BAS, test_size=0.25, random_state=42)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
            y_pred = predict_fn(clf, X_test)
            accs.append(np.mean(y_pred == y_test))
        elif dataset == 'nsphere':
            D_circle = 3
            N_circle = 1000
            r1, r0 = 1, 0.5
            dr = 0.2
            r_mid = (r1 + r0) / 2
            x_circle1 = r1 * Noisy_nsphere_sample(dr / r1, int(N_circle / 2), ndim=D_circle).T
            x_circle0 = r0 * Noisy_nsphere_sample(dr / r0, int(N_circle / 2), ndim=D_circle).T
            x_circle = np.dstack((x_circle1.T, x_circle0.T)).reshape(D_circle, N_circle).T
            rad_circle = np.sqrt(np.sum(x_circle ** 2, axis=1))
            Y_circle = (rad_circle > r_mid).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(x_circle, Y_circle, test_size=0.25, random_state=42+run, stratify=Y_circle)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
            y_pred = predict_fn(clf, X_test)
            accs.append(np.mean(y_pred == y_test))
        elif dataset == 'circles':
            D_circle = 2
            N_circle = 1000
            r1, r0 = 1, 0.5
            dr = 0.2
            r_mid = (r1 + r0) / 2
            x_circle1 = r1 * Noisy_nsphere_sample(dr / r1, int(N_circle / 2), ndim=D_circle).T
            x_circle0 = r0 * Noisy_nsphere_sample(dr / r0, int(N_circle / 2), ndim=D_circle).T
            x_circle = np.dstack((x_circle1.T, x_circle0.T)).reshape(D_circle, N_circle).T
            rad_circle = np.sqrt(np.sum(x_circle ** 2, axis=1))
            Y_circle = (rad_circle > r_mid).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(x_circle, Y_circle, test_size=0.25, random_state=42+run, stratify=Y_circle)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
            y_pred = predict_fn(clf, X_test)
            accs.append(np.mean(y_pred == y_test))
        elif dataset == 'spiral':
            D_spiral = 2
            N_spiral = 1000
            cphase = [0, 1]
            Nturns = 4
            Sep = 0.05
            W = 0.5
            x_spiral0 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep).T
            x_spiral1 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep).T
            x_spiral = np.dstack((x_spiral1.T, x_spiral0.T)).reshape(D_spiral, N_spiral).T
            Y_spiral = np.dstack((np.ones(int(N_spiral / 2)), np.zeros(int(N_spiral / 2)))).flatten().astype(int)
            X_train, X_test, y_train, y_test = train_test_split(x_spiral, Y_spiral, test_size=0.25, random_state=42+run, stratify=Y_spiral)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
            y_pred = predict_fn(clf, X_test)
            accs.append(np.mean(y_pred == y_test))
        elif dataset == 'spiral2':
            D_spiral = 2
            N_spiral = 1000
            cphase = [0, 1]
            Nturns = 4
            Sep = 0.05
            W = 0.5
            x_spiral0 = Spiral_sample(W, int(N_spiral / 2), ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep).T
            x_spiral1 = Spiral_sample(W, int(N_spiral / 2), ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep).T
            x_spiral = np.dstack((x_spiral1.T, x_spiral0.T)).reshape(D_spiral, N_spiral).T
            Y_spiral = np.dstack((np.ones(int(N_spiral / 2)), np.zeros(int(N_spiral / 2)))).flatten().astype(int)
            X_train, X_test, y_train, y_test = train_test_split(x_spiral, Y_spiral, test_size=0.25, random_state=42+run, stratify=Y_spiral)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
            y_pred = predict_fn(clf, X_test)
            accs.append(np.mean(y_pred == y_test))
        else:
            print(f"Unknown dataset: {dataset}")
            return
    accs = np.array(accs)
    print(f"{dataset.capitalize()} Test accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}% (std over {n_runs} runs)")

def main():
    parser = argparse.ArgumentParser(description="Train and test SVM on basic datasets.")
    parser.add_argument('--dataset', type=str, default='bas', 
                        choices=['bas', 'nsphere', 'spiral', 'spiral2', 'circles'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='classical_svm', 
                    choices=['classical_svm', 'feedforward', 'transformer', 'quantum_kernel', 'quantum_vc1'], 
                    help='Model to use')
    args = parser.parse_args()
    train_and_test(args.dataset, args.model)

if __name__ == "__main__":
    """
     if you want to use residual-block for FeedForwardNet, 
     modify the FeedForwardNet class in models/feed_forward_net.py accordingly

     Same with kernel choice for SVM in models/classical_svm.py, we can
     use 'rbf' kernel instead of 'linear' by default
    """
    main()