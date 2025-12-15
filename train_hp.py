import numpy as np
import argparse
import itertools
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from basic_datasets import get_bas_example, nsphere_sample, Spiral_sample, Spiral_sample2, Noisy_nsphere_sample, get_noisy_bas_example
from models.classical_svm import fit_svm_classifier, predict_svm_classifier
from models.feed_forward_net import fit_feedforward_classifier, predict_feedforward_classifier
from models.transformer import fit_transformer_classifier, predict_transformer_classifier
from models.quantum_kernel import fit_quantumkernel_classifier, predict_quantumkernel_classifier
from models.quantum_hybrid import fit_hybrid_qnn_classifier, predict_hybrid_qnn_classifier
from models.quantum_vc1 import train_vqc1, predict_vqc1

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

MODELS = {
    'classical_svm': (fit_svm_classifier, predict_svm_classifier),
    "feedforward": (fit_feedforward_classifier, predict_feedforward_classifier),
    'transformer': (fit_transformer_classifier, predict_transformer_classifier),
    'quantum_kernel': (fit_quantumkernel_classifier, predict_quantumkernel_classifier),
    'quantum_vc1': (train_vqc1, predict_vqc1),
    'hybrid_qnn': (fit_hybrid_qnn_classifier, predict_hybrid_qnn_classifier),
}

PARAM_GRIDS = {
    'classical_svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 1],
        'kernel': ['rbf']  # Force RBF as discussed
    },
    'feedforward': {
        'lr': [0.001, 0.01, 0.05],
        'hidden_dim': [16, 32, 64],
        'epochs': [50, 100]
    },
    'transformer': {
        'lr': [0.001, 0.01],
        'd_model': [4, 8],
        'num_layers': [1, 2],
        'epochs': [50]
    },
    'quantum_kernel': {
        'C': [0.1, 1, 10, 100]
    },
    'hybrid_qnn': {
        'lr': [0.01, 0.05],
        'n_q_layers': [2, 3],
        'epochs': [30]
    },
    'quantum_vc1': {
        'n_layers': [1, 2, 3],
        'maxiter': [50, 100, 150]
    }
}

def get_data_for_run(dataset: str, run_idx: int):
    """
    Generates fresh data for a specific run index.
    Ensures that Spiral datasets are seeded dynamically (100 + run) so we test on new data,
    while BAS remains constant and NSphere uses global random state.
    """
    if dataset == 'bas':
        x_BAS, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary = get_bas_example()
        return x_BAS, Y_BAS, False  # False = do not scale BAS
    
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
        return x_circle, Y_circle, True

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
        return x_circle, Y_circle, True

    elif dataset == 'spiral':
        D_spiral = 2
        N_spiral = 1000
        cphase = [0, 1]
        Nturns = 4
        Sep = 0.05
        W = 0.5
        # Use Spiral_sample (standard) and seed dynamically
        x_spiral0 = Spiral_sample(W, int(N_spiral / 2), ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep, seed=100+run_idx).T
        x_spiral1 = Spiral_sample(W, int(N_spiral / 2), ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep, seed=100+run_idx).T
        x_spiral = np.dstack((x_spiral1.T, x_spiral0.T)).reshape(D_spiral, N_spiral).T
        Y_spiral = np.dstack((np.ones(int(N_spiral / 2)), np.zeros(int(N_spiral / 2)))).flatten().astype(int)
        return x_spiral, Y_spiral, True

    elif dataset == 'spiral2':
        D_spiral = 2
        N_spiral = 1000
        cphase = [0, 1]
        Nturns = 4
        Sep = 0.05
        W = 0.5
        # Use Spiral_sample2 (noisy/radial) and seed dynamically
        x_spiral0 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep, seed=100+run_idx).T
        x_spiral1 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep, seed=100+run_idx).T
        x_spiral = np.dstack((x_spiral1.T, x_spiral0.T)).reshape(D_spiral, N_spiral).T
        Y_spiral = np.dstack((np.ones(int(N_spiral / 2)), np.zeros(int(N_spiral / 2)))).flatten().astype(int)
        return x_spiral, Y_spiral, True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def train_and_test(dataset: str, model: str):
    if model not in MODELS:
        print(f"Model '{model}' not implemented yet. Use existing models.")
        return
    fit_fn, predict_fn = MODELS[model]
    n_runs = 5
    accs = []

    for run in range(n_runs):
        try:
            X_raw, Y_raw, use_scaler = get_data_for_run(dataset, run)
        except ValueError as e:
            print(e)
            return

        # split data: 20% testing, 20% validation, 60% training
        X_temp, X_test_raw, y_temp, y_test = train_test_split(
            X_raw, Y_raw, test_size=0.2, random_state=42+run, stratify=Y_raw
        )
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42+run, stratify=y_temp
        )

        if use_scaler:
            scaler_tune = StandardScaler()
            X_train_tune = scaler_tune.fit_transform(X_train_raw)
            X_val_tune = scaler_tune.transform(X_val_raw)
        else:
            X_train_tune, X_val_tune = X_train_raw, X_val_raw

        # hyperparameter tuning
        best_val_acc = -1.0
        best_params = {}
        best_clf = None

        grid = PARAM_GRIDS.get(model, {})
        keys, values = zip(*grid.items()) if grid else ([], [])
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Run {run}, tuning {len(combinations)} configs for {model}")

        for params in combinations:
            current_clf = fit_fn(X_train_tune, y_train, **params)
            val_preds = predict_fn(current_clf, X_val_tune)
            val_acc = np.mean(val_preds == y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
                best_clf = best_params
        
        print(f"Best validation accuracy: {best_val_acc:.4f}, hyperparameters: {best_clf}")

        # refit with best hyperparameters
        X_combined_raw = np.concatenate([X_train_raw, X_val_raw])
        y_combined = np.concatenate([y_train, y_val])

        if use_scaler:
            scaler_final = StandardScaler()
            X_combined_final = scaler_final.fit_transform(X_combined_raw)
            X_test_final = scaler_final.transform(X_test_raw)
        else:
            X_combined_final = X_combined_raw
            X_test_final = X_test_raw

        clf = fit_fn(X_combined_final, y_combined, **best_params)
            
        if run == 0 and hasattr(clf, 'parameters'):
            n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {n_params}")

        y_pred = predict_fn(clf, X_test_final)

        if run == 0:
            print(f"Classification Report (Run {run})")
            print(classification_report(y_test, y_pred))

        accs.append(np.mean(y_pred == y_test))

    accs = np.array(accs)
    print(f"{dataset.capitalize()} Test accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}% (std over {n_runs} runs)")

def main():
    parser = argparse.ArgumentParser(description="Train and test SVM on basic datasets.")
    parser.add_argument('--dataset', type=str, default='bas', 
                        choices=['bas', 'nsphere', 'spiral', 'spiral2', 'circles'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='classical_svm', 
                    choices=['classical_svm', 'feedforward', 'transformer', 'quantum_kernel', 'quantum_vc1', 'hybrid_qnn'], 
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
    # print(torch.backends.mps.is_available())
    # raise Exception
    main()
    