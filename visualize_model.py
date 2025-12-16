import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from basic_datasets import get_bas_example, nsphere_sample, Spiral_sample, Spiral_sample2
from models.classical_svm import fit_svm_classifier, predict_svm_classifier
from models.quantum_kernel import fit_quantumkernel_classifier, predict_quantumkernel_classifier
from models.quantum_hybrid import fit_hybrid_qnn_classifier, predict_hybrid_qnn_classifier
from models.quantum_vc1 import train_vqc1, predict_vqc1


def load_2d_dataset(name: str):
    """Return (X, y) for a 2D dataset name in {bas, nsphere, circles, spiral, spiral2}."""
    if name == "bas":
        X, y, *_ = get_bas_example()
        return X[:, :2], y  # take first 2 dims for visualization
    elif name == "nsphere":
        # nsphere_sample returns coordinates of shape (ndim, N) on the unit sphere, without labels.
        # For visualization, build a simple two-class problem using radius with a bit of noise.
        ndim = 2
        N_inner = 500
        N_outer = 500
        # Inner noisy points around radius 0.5
        r_inner = 0.5 + 0.05 * np.random.randn(N_inner)
        theta_inner = 2 * np.pi * np.random.rand(N_inner)
        X_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)
        y_inner = np.zeros(N_inner, dtype=int)
        # Outer noisy points around radius 1.0
        r_outer = 1.0 + 0.05 * np.random.randn(N_outer)
        theta_outer = 2 * np.pi * np.random.rand(N_outer)
        X_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)
        y_outer = np.ones(N_outer, dtype=int)
        X = np.vstack([X_inner, X_outer])
        y = np.concatenate([y_inner, y_outer])
        return X, y
    elif name == "spiral":
        D_spiral = 2
        N_spiral = 1000
        cphase = [0, 1]
        Nturns = 4
        Sep = 0.05
        W = 0.5
        x0 = Spiral_sample2(W, N_spiral // 2, ts=cphase[0] * np.pi, Nturns=Nturns, Sep=Sep).T
        x1 = Spiral_sample2(W, N_spiral // 2, ts=cphase[1] * np.pi, Nturns=Nturns, Sep=Sep).T
        X = np.dstack((x1.T, x0.T)).reshape(D_spiral, N_spiral).T
        y = np.dstack((np.ones(N_spiral // 2), np.zeros(N_spiral // 2))).flatten().astype(int)
        return X, y
    elif name == "spiral2":
        X, y = Spiral_sample(0.75, 1000)
        return X.T, y
    else:
        raise ValueError(f"Unknown dataset: {name}")


def plot_decision_boundary(model_predict, X, y, title="Decision Boundary", grid_step=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Ensure results directory exists and save figure there
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{title.replace(' ', '_')}.png"
    plt.title(title)
    plt.savefig(os.path.join(results_dir, fname), dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize decision boundaries for models on toy datasets.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["classical_svm", "quantum_kernel", "quantum_hybrid", "quantum_vc1"],
                        help="Model to visualize")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["bas", "nsphere", "spiral", "spiral2"],
                        help="Dataset to use")
    args = parser.parse_args()

    X, y = load_2d_dataset(args.dataset)
    print(X.shape, y.shape)

    if args.model == "classical_svm":
        clf = fit_svm_classifier(X, y)
        model_predict = lambda Xg: predict_svm_classifier(clf, Xg)
    elif args.model == "quantum_kernel":
        clf = fit_quantumkernel_classifier(X, y)
        model_predict = lambda Xg: predict_quantumkernel_classifier(clf, Xg)
    elif args.model == 'quantum_vc1':
        modelw = train_vqc1(X, y, epochs=50)
        model_predict = lambda Xg: predict_vqc1(modelw, Xg)
    else:
        modelw = fit_hybrid_qnn_classifier(X, y, head="quantum_only", n_measurements=3, epochs=50)
        model_predict = lambda Xg: predict_hybrid_qnn_classifier(modelw, Xg)

    plot_decision_boundary(model_predict, X, y,
                           title=f"{args.model} on {args.dataset}")


if __name__ == "__main__":
    main()
