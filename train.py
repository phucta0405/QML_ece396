import numpy as np
from sklearn.model_selection import train_test_split
from basic_datasets import get_bas_example, nsphere_sample, Spiral_sample, Spiral_sample2
from models.classical_svm import fit_svm_classifier, predict_svm_classifier

DATASET_LOADERS = {
    'bas': get_bas_example,
    'nsphere': lambda: nsphere_sample(200, ndim=2),
    'spiral': lambda: Spiral_sample(dW=0.1, Ns=200),
    'spiral2': lambda: Spiral_sample2(dW=0.1, Ns=200),
}

def train_and_test_bas():
    x_BAS, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary = get_bas_example()
    X_train, X_test, y_train, y_test = train_test_split(x_BAS, Y_BAS, test_size=0.25, random_state=42)
    clf = fit_svm_classifier(X_train, y_train, kernel='linear', C=1.0)
    y_pred = predict_svm_classifier(clf, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"BAS Test accuracy: {accuracy * 100:.2f}%")


def train_and_test(dataset: str):
    if dataset == 'bas':
        train_and_test_bas()
    elif dataset == 'nsphere':
        X = nsphere_sample(200, ndim=2).T
        Y = np.zeros(X.shape[0])  # Dummy labels, replace with actual
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        clf = fit_svm_classifier(X_train, y_train)
        y_pred = predict_svm_classifier(clf, X_test)
        print(f"Nsphere Test accuracy: {np.mean(y_pred == y_test) * 100:.2f}%")
    elif dataset == 'spiral':
        X = Spiral_sample(dW=0.1, Ns=200).T
        Y = np.zeros(X.shape[0])  # Dummy labels, replace with actual
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        clf = fit_svm_classifier(X_train, y_train)
        y_pred = predict_svm_classifier(clf, X_test)
        print(f"Spiral Test accuracy: {np.mean(y_pred == y_test) * 100:.2f}%")
    else:
        print(f"Unknown dataset: {dataset}")


def main():
    train_and_test('bas')

if __name__ == "__main__":
    main()