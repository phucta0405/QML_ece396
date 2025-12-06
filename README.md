# Quantum Machine Learning with SVM
Quantum Machine Learning projects by **Steve Ta**[^eq][^cs], **Coco Gong**[^eq][^cs], and **Sasha Cocquyt**[^eq][^ece].

[^eq]: Equal contributions.
[^cs]: Princeton Department of Computer Science.
[^ece]: Princeton Department of Electrical and Computer Engineering.

## How to Use `train.py`

This project provides a command-line interface for training and testing classical SVMs on several toy datasets. You can select the dataset and model from the terminal.

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run training and testing

Use the following command to run the training script:

```bash
python3 train.py --dataset DATASET_NAME --model MODEL_NAME
```

- `--dataset` can be one of: `bas`, `nsphere`, `circles`, `spiral`
- `--model` is currently only `classical_svm` (future models can be added)

#### Examples

Train and test on Bars and Stripes:
```bash
python3 train.py --dataset bas --model classical_svm
```

Train and test on noisy circles:
```bash
python3 train.py --dataset circles --model classical_svm
```

Train and test on spiral dataset:
```bash
python3 train.py --dataset spiral --model classical_svm
```

Train and test on n-sphere dataset:
```bash
python3 train.py --dataset nsphere --model classical_svm
```

### 3. Output

The script will print the test accuracy for the selected dataset and model.

---
