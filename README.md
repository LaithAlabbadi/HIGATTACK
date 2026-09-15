# HIGAttack Evaluation Pipeline

This repository contains the PyTorch implementation of HIGAttack, an adversarial attack utilizing an Integrated Hessian interaction matrix and a graph Laplacian penalty. The pipeline evaluates the robustness of a security-focused Multi-Layer Perceptron (MLP) against this attack and standard baselines (FGSM, PGD, JSMA).

The script is currently configured to evaluate against the UNSW-NB15 intrusion detection dataset.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline via the command line. You can specify one or multiple random seeds for reproducible runs.

```bash
python main.py --seeds 11,22,33,44,55
```

The script automatically downloads the UNSW-NB15 dataset via `kagglehub` if it is not present in the local cache.

## Pipeline Details

1. **Hyperparameter Tuning:** Performs a grid search over learning rates and momentum for a standard SGD optimizer.
2. **Model Checkpointing:** Saves the best performing model based on validation accuracy.
3. **Attack Evaluation:** Runs standard attacks (FGSM, PGD, JSMA) and the proposed HIGAttack (including ablations).
4. **Robustness Checkpointing:** Automatically saves evaluation progress. If the script is interrupted, re-running the same command will resume from the last processed batch.

## Outputs

The script creates the following directories in the working directory:
* `models/`: Saved `*.pth` model weights.
* `checkpoints/`: State files for the robustness evaluation loop.
* `results/`: JSON files containing aggregated metrics (Attack Success Rate, Accuracy, Precision/Recall/F1, ECE, and Lp-norm distortions).
