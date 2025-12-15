import numpy as np
import argparse
import random
import torch
import wandb
from sklearn.model_selection import train_test_split
from basic_datasets import get_bas_example, nsphere_sample, Spiral_sample, Spiral_sample2, Noisy_nsphere_sample, get_noisy_bas_example
from models.classical_svm import fit_svm_classifier, predict_svm_classifier
from models.feed_forward_net import fit_feedforward_classifier, predict_feedforward_classifier
from models.transformer import fit_transformer_classifier, predict_transformer_classifier
from models.quantum_kernel import fit_quantumkernel_classifier, predict_quantumkernel_classifier
from models.quantum_vc import train_vqc, predict_vqc  
from quantum_vc1 import train_vqc1, predict_vqc1
from models.quantum_hybrid import fit_hybrid_qnn_classifier, predict_hybrid_qnn_classifier
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
    'spiral': lambda: Spiral_sample(dW=0.75, Ns=number_of_samples),
    'spiral2': lambda: Spiral_sample2(dW=0.75, Ns=number_of_samples),
}

MODELS = {
    #'classical_svm': (fit_svm_classifier, predict_svm_classifier),
    #"feedforward": (fit_feedforward_classifier, predict_feedforward_classifier),
    'transformer': (fit_transformer_classifier, predict_transformer_classifier),
    #'quantum_kernel': (fit_quantumkernel_classifier, predict_quantumkernel_classifier),
    #'quantum_vc1': (train_vqc1, predict_vqc1),
    #'hybrid_qnn': (fit_hybrid_qnn_classifier, predict_hybrid_qnn_classifier),
    #'quantum_vc': (train_vqc, predict_vqc), 
}

def train_and_test(dataset: str, model: str):
    if model not in MODELS:
        print(f"Model '{model}' not implemented yet. Use existing models.")
        return

    # Initialize wandb
    wandb.init(
        project="qml-classification",
        config={
            "dataset": dataset,
            "model": model,
            "n_runs": 5,
            "number_of_samples": number_of_samples,
        }
    )

    fit_fn, predict_fn = MODELS[model]
    n_runs = 1
    accs = []
    for run in range(n_runs):
        if dataset == 'bas':
            x_BAS, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary = get_bas_example()
            X_train, X_test, y_train, y_test = train_test_split(x_BAS, Y_BAS, test_size=0.25, random_state=42)
            clf = fit_fn(X_train, y_train)
            if run == 0 and hasattr(clf, 'parameters'):
                n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {n_params}")
                wandb.config.update({"n_parameters": n_params})
            y_pred = predict_fn(clf, X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)
            wandb.log({"run": run, "accuracy": acc})
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
                wandb.config.update({"n_parameters": n_params})
            y_pred = predict_fn(clf, X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)
            wandb.log({"run": run, "accuracy": acc})
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
                wandb.config.update({"n_parameters": n_params})
            y_pred = predict_fn(clf, X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)
            wandb.log({"run": run, "accuracy": acc})
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
                wandb.config.update({"n_parameters": n_params})
            y_pred = predict_fn(clf, X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)
            wandb.log({"run": run, "accuracy": acc})
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
                wandb.config.update({"n_parameters": n_params})
            y_pred = predict_fn(clf, X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)
            wandb.log({"run": run, "accuracy": acc})
        else:
            print(f"Unknown dataset: {dataset}")
            return
    accs = np.array(accs)
    mean_acc = accs.mean()
    std_acc = accs.std()

    # Log summary statistics
    wandb.log({
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_accuracy_percent": mean_acc * 100,
        "std_accuracy_percent": std_acc * 100
    })

    print(f"{dataset.capitalize()} Test accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (std over {n_runs} runs)")

    # Finish wandb run
    wandb.finish()

def sweep_spiral_dim(models, min_dim=2, max_dim=10, step=2, num_samples=10000, n_runs=1):
    """
    For each model, test on spiral dataset with increasing dimension and log to wandb.
    Log a single plot with all models' accuracy curves.
    """
    all_mean_accs = {}
    dims = list(range(min_dim, max_dim+1, step))
    for model in models:
        wandb.init(
            project="qml-classification",
            config={
                "sweep_type": "spiral_dim",
                "model": model,
                "min_dim": min_dim,
                "max_dim": max_dim,
                "step": step,
                "num_samples": num_samples,
                "n_runs": n_runs,
            },
            name=f"sweep_{model}_spiral_dim",
            reinit=True
        )
        fit_fn, predict_fn = MODELS[model]
        mean_accs = []
        std_accs = []
        for d in dims:
            accs = []
            for run in range(n_runs):
                if d == 2:
                    # Match train.py spiral 
                    D_spiral = 2
                    N_spiral = num_samples
                    cphase = [0, 1]
                    Nturns = 4
                    Sep = 0.05
                    W = 0.5
                    x_spiral0 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep).T
                    x_spiral1 = Spiral_sample2(W, int(N_spiral / 2), ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep).T
                    x_spiral = np.dstack((x_spiral1.T, x_spiral0.T)).reshape(D_spiral, N_spiral).T
                    y_spiral = np.dstack((np.ones(int(N_spiral / 2)), np.zeros(int(N_spiral / 2)))).flatten().astype(int)
                else:
                    # For d>2, stack 2D spiral with random noise in extra dims
                    x_spiral2d = Spiral_sample2(0.75, num_samples, ts=0, Nturns=4, Sep=0.05, seed=42+run).T
                    extra = np.random.randn(num_samples, d-2)
                    x_spiral = np.concatenate([x_spiral2d, extra], axis=1)
                    y_spiral = np.dstack((np.zeros(num_samples//2), np.ones(num_samples//2))).flatten().astype(int)
                X_train, X_test, y_train, y_test = train_test_split(x_spiral, y_spiral, test_size=0.25, random_state=42+run, stratify=y_spiral)
                clf = fit_fn(X_train, y_train)
                y_pred = predict_fn(clf, X_test)
                acc = np.mean(y_pred == y_test)
                accs.append(acc)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
            wandb.log({
                f"{model}_dim": d,
                f"{model}_mean_accuracy": mean_acc,
                f"{model}_std_accuracy": std_acc
            })
        all_mean_accs[model] = mean_accs
        wandb.finish()
    # Log a single plot with all models' accuracy curves
    wandb.init(project="qml-classification_spiral_noise", name="spiral_dim_all_models_plot", reinit=True)
    ys = [all_mean_accs[m] for m in models]
    keys = [f"{m} mean accuracy" for m in models]
    wandb.log({
        "all_models_acc_vs_dim": wandb.plot.line_series(
            xs=dims,
            ys=ys,
            keys=keys,
            title="All Models: Accuracy vs. Dimension (Spiral Dataset)",
            xname="dimension"
        )
    })
    wandb.finish()

def sweep_nsphere_dim(models, min_dim=2, max_dim=10, step=2, num_samples=10000, n_runs=1):
    """
    For each model, test on nsphere dataset with increasing dimension and log to wandb.
    Log a single plot with all models' accuracy curves.
    """
    all_mean_accs = {}
    dims = list(range(min_dim, max_dim+1, step))
    for model in models:
        wandb.init(
            project="qml-classification",
            config={
                "sweep_type": "nsphere_dim",
                "model": model,
                "min_dim": min_dim,
                "max_dim": max_dim,
                "step": step,
                "num_samples": num_samples,
                "n_runs": n_runs,
            },
            name=f"sweep_{model}_nsphere_dim",
            reinit=True
        )
        fit_fn, predict_fn = MODELS[model]
        mean_accs = []
        std_accs = []
        for d in dims:
            accs = []
            for run in range(n_runs):
                # Generate nsphere data in d dimensions
                r1, r0 = 1, 0.5
                dr = 0.2
                r_mid = (r1 + r0) / 2
                x_sphere1 = r1 * Noisy_nsphere_sample(dr / r1, int(num_samples / 2), ndim=d).T
                x_sphere0 = r0 * Noisy_nsphere_sample(dr / r0, int(num_samples / 2), ndim=d).T
                x_sphere = np.vstack([x_sphere1, x_sphere0])
                rad_sphere = np.sqrt(np.sum(x_sphere ** 2, axis=1))
                Y_sphere = (rad_sphere > r_mid).astype(int)
                X_train, X_test, y_train, y_test = train_test_split(x_sphere, Y_sphere, test_size=0.25, random_state=42+run, stratify=Y_sphere)
                clf = fit_fn(X_train, y_train)
                y_pred = predict_fn(clf, X_test)
                acc = np.mean(y_pred == y_test)
                accs.append(acc)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
            wandb.log({
                f"{model}_dim": d,
                f"{model}_mean_accuracy": mean_acc,
                f"{model}_std_accuracy": std_acc
            })
        all_mean_accs[model] = mean_accs
        wandb.finish()
    # Log a single plot with all models' accuracy curves
    wandb.init(project="qml-classification-nsphere", name="nsphere_dim_all_models_plot", reinit=True)
    ys = [all_mean_accs[m] for m in models]
    keys = [f"{m} mean accuracy" for m in models]
    wandb.log({
        "all_models_acc_vs_dim_nsphere": wandb.plot.line_series(
            xs=dims,
            ys=ys,
            keys=keys,
            title="All Models: Accuracy vs. Dimension (n-Sphere Dataset)",
            xname="dimension"
        )
    })
    wandb.finish()

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
    # Example usage: sweep all models on spiral dataset with increasing dimension
    sweep_spiral_dim(list(MODELS.keys()), min_dim=2, max_dim=14, step=1, num_samples=1000, n_runs=1)