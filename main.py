import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import glob
import random
import kagglehub
import matplotlib.pyplot as plt
from datetime import datetime
from torch.autograd.functional import hessian
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# == == == == == == == == == == == == == == == == == == == == ==
#
# 1. MODEL ARCHITECTURE
#
# == == == == == == == == == == == == == == == == == == == == ==

class SecurityMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SecurityMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Softplus(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# == == == == == == == == == == == == == == == == == == == == ==
#
# 2. ADVERSARIAL ATTACKS
#
# == == == == == == == == == == == == == == == == == == == == ==

def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, y)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(model, x, y, epsilon, alpha=0.01, num_iter=50, random_start=True):
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)

    x_adv.requires_grad_(True)
    for _ in range(num_iter):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = (x + eta).detach().requires_grad_(True)
    return x_adv.detach()


def jsma_attack(model, x, y, theta=1.0, gamma=0.1, clip_min=None, clip_max=None):
    model.eval()
    x_adv = x.clone().detach()
    batch_size, num_features = x.shape
    max_iters = int(num_features * gamma)

    for b in range(batch_size):
        sample = x_adv[b:b+1].clone().detach()
        sample.requires_grad_(True)
        target = (1 - y[b]).item()

        for _ in range(max_iters):
            logits = model(sample)
            pred = logits.argmax(dim=1).item()
            if pred == target:
                break

            jacobian = []
            for cls in range(logits.shape[1]):
                grad_outputs = torch.zeros_like(logits)
                grad_outputs[0, cls] = 1.0
                grads = torch.autograd.grad(
                    outputs=logits,
                    inputs=sample,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False
                )[0]
                jacobian.append(grads)

            jacobian = torch.stack(jacobian, dim=0)
            grad_t = jacobian[target, 0]
            other_classes = [c for c in range(logits.shape[1]) if c != target]
            grad_others = jacobian[other_classes, 0].sum(dim=0)

            alpha = grad_t
            beta = grad_others
            saliency = torch.zeros_like(alpha)
            mask = (alpha > 0) & (beta < 0)
            saliency[mask] = alpha[mask] * torch.abs(beta[mask])

            if torch.all(saliency == 0):
                break

            feature_idx = torch.argmax(saliency).item()
            with torch.no_grad():
                sample[0, feature_idx] += theta * torch.sign(grad_t[feature_idx])
                if clip_min is not None or clip_max is not None:
                    sample = torch.clamp(sample, clip_min, clip_max)
            sample.requires_grad_(True)
        x_adv[b] = sample.detach()
    return x_adv


# --- New Attack Components ---

def compute_hessian(model, x, target):
    def score(inp):
        logits = model(inp.unsqueeze(0))
        return torch.log_softmax(logits, dim=1)[0, target]
    H = hessian(score, x.squeeze())
    return H


def integrated_hessian_2d(model, x, baseline, target, steps=20):
    """
    Compute the Integrated Hessian interaction matrix.

    Parameters
    ----------
    model : nn.Module
    x : Tensor
        Input tensor (same shape as model input).
    baseline : Tensor
        Baseline input.
    target : int
        Target class/logit.
    steps : int
        Number of Riemann integration steps.

    Returns
    -------
    Tensor
        Integrated Hessian interaction matrix (d x d).
    """

    diff = (x - baseline).view(-1)
    d = diff.numel()

    interaction = torch.zeros(d, d, device=x.device)

    for a in range(1, steps + 1):
        alpha = a / steps

        for b in range(1, steps + 1):
            beta = b / steps

            # x(alpha,beta) = x0 + alpha*beta*(x-x0)
            point = baseline + (alpha * beta) * (x - baseline)
            point = point.detach().clone().requires_grad_(True)

            # Hessian wrt input
            H = compute_hessian(model, point, target)

            # αβ weighting from the derivation
            interaction += (alpha * beta) * H

    # Approximate double integral
    interaction /= (steps * steps)

    # Multiply by (x-x0)(x-x0)^T
    interaction *= torch.outer(diff, diff)

    return interaction


def build_adjacency(interaction):
    A = interaction.abs()
    A.fill_diagonal_(0.)
    return A


def symmetric_normalized_laplacian(A):
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L_sym = torch.eye(A.size(0), device=A.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    return L_sym


def generate_attack(model, x, baseline, target, eps=0.1, lam=0.1, num_steps=50, use_integrated=True):
    model.eval()
    x_in = x.detach().clone()
    
    # 1. Targeted Attack: We want the model to actively predict the 'target' class
    target_label = torch.tensor([target], dtype=torch.long, device=x.device)
    
    # 2. Compute Laplacian based on Hessian type if lam > 0
    if lam > 0:
        if use_integrated:
            IH = integrated_hessian_2d(model, x_in.unsqueeze(0), baseline.unsqueeze(0), target)
        else:
            # Local Hessian ablation
            H = compute_hessian(model, x_in, target)
            diff = (x_in - baseline).view(-1)
            IH = H * torch.outer(diff, diff)
            
        A = build_adjacency(IH)
        L = symmetric_normalized_laplacian(A)
    else:
        L = None
    
    # 3. Initialize delta as a leaf tensor and use Adam Optimizer 
    delta = torch.zeros_like(x_in, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.02) 
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        
        x_adv = x_in + delta
        logits = model(x_adv.unsqueeze(0))
        
        # Loss 1: Targeted Classification Loss
        loss_cls = F.cross_entropy(logits, target_label)
        
        # Loss 2: Laplacian Quadratic Penalty
        if L is not None:
            loss_graph = torch.dot(delta, torch.mv(L, delta))
            total_loss = loss_cls + lam * loss_graph
        else:
            total_loss = loss_cls
            
        total_loss.backward()
        optimizer.step()
        
        # Project back to L-infinity ball
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
            
    x_adv = (x_in + delta).detach()
    return x_adv


# == == == == == == == == == == == == == == == == == == == == ==
#
# 3. TRAINING & EVALUATION
#
# == == == == == == == == == == == == == == == == == == == == ==

def atomic_torch_save(value, path):
    """Write a PyTorch file atomically so interruption cannot corrupt it."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    temp_path = f"{path}.tmp"
    torch.save(value, temp_path)
    os.replace(temp_path, path)


def save_model(model, path):
    atomic_torch_save(model.state_dict(), path)


def load_model(model, path):
    model_device = next(model.parameters()).device
    model.load_state_dict(torch.load(path, map_location=model_device))


def evaluate_clean(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return {'Acc': correct / total}


def train_baseline(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def compute_ece(logits, labels, n_bins=10):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def compute_distortions(x_clean, x_adv, threshold=1e-4):
    delta = x_adv - x_clean
    return {
        "L0": (delta.abs() > threshold).float().sum(dim=1).cpu(),
        "L2": torch.norm(delta, p=2, dim=1).cpu(),
        "Linf": torch.norm(delta, p=float("inf"), dim=1).cpu()
    }


def evaluate_robustness(model, loader, epsilon, device, checkpoint_path=None):
    model.eval()
    all_labels = []
    preds = {"Clean": [], "FGSM": [], "PGD": [], "JSMA": [], "NewAttack": [], "NewAttack_NoLap": [], "NewAttack_LocalLap": []}
    probs = {"Clean": [], "FGSM": [], "PGD": [], "JSMA": [], "NewAttack": [], "NewAttack_NoLap": [], "NewAttack_LocalLap": []}
    all_logits = []

    distortions = {
        "FGSM": {"L0": [], "L2": [], "Linf": []},
        "PGD": {"L0": [], "L2": [], "Linf": []},
        "JSMA": {"L0": [], "L2": [], "Linf": []},
        "NewAttack": {"L0": [], "L2": [], "Linf": []},
        "NewAttack_NoLap": {"L0": [], "L2": [], "Linf": []},
        "NewAttack_LocalLap": {"L0": [], "L2": [], "Linf": []},
    }

    total_samples = len(loader.dataset)
    processed_samples = 0
    resume_rng_state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("total_samples") != total_samples:
            raise ValueError(
                f"Checkpoint expects {checkpoint.get('total_samples')} samples, "
                f"but the current test set has {total_samples}"
            )
        if not np.isclose(checkpoint.get("epsilon"), epsilon):
            raise ValueError(
                f"Checkpoint epsilon {checkpoint.get('epsilon')} does not match {epsilon}"
            )
        processed_samples = checkpoint["processed_samples"]
        all_labels = checkpoint["all_labels"]
        preds = checkpoint["preds"]
        probs = checkpoint["probs"]
        all_logits = checkpoint["all_logits"]
        distortions = checkpoint["distortions"]
        resume_rng_state = checkpoint["rng_state"]
        print(
            f"    Resuming robustness evaluation at sample "
            f"{processed_samples}/{total_samples}..."
        )

    resume_samples = processed_samples
    skipped_samples = 0
    rng_restored = resume_rng_state is None
    for x, y in loader:
        batch_size = x.size(0)
        if skipped_samples < resume_samples:
            if skipped_samples + batch_size > resume_samples:
                raise ValueError("Checkpoint position is not aligned with the test batches")
            skipped_samples += batch_size
            continue

        # Constructing/advancing a DataLoader iterator can consume the global
        # PyTorch RNG. Restore the saved state only after resume rows are skipped.
        if not rng_restored:
            torch.set_rng_state(resume_rng_state["torch"])
            np.random.set_state(resume_rng_state["numpy"])
            random.setstate(resume_rng_state["python"])
            if torch.cuda.is_available() and resume_rng_state["cuda"] is not None:
                torch.cuda.set_rng_state_all(resume_rng_state["cuda"])
            rng_restored = True

        x, y = x.to(device), y.to(device)
        all_labels.append(y.cpu().numpy())

        with torch.no_grad():
            outputs = model(x)
            all_logits.append(outputs.cpu())
            clean_preds = outputs.max(1)[1].cpu().numpy()
            preds["Clean"].append(clean_preds)
            probs["Clean"].append(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        # Standard attacks
        x_fgsm = fgsm_attack(model, x, y, epsilon)
        x_pgd = pgd_attack(model, x, y, epsilon)
        
        with torch.no_grad():
            out_fgsm = model(x_fgsm)
            preds["FGSM"].append(out_fgsm.max(1)[1].cpu().numpy())
            probs["FGSM"].append(F.softmax(out_fgsm, dim=1)[:, 1].cpu().numpy())

            out_pgd = model(x_pgd)
            preds["PGD"].append(out_pgd.max(1)[1].cpu().numpy())
            probs["PGD"].append(F.softmax(out_pgd, dim=1)[:, 1].cpu().numpy())

        # Slower attacks
        x_jsma = jsma_attack(model, x, y)
        
        with torch.no_grad():
            out_jsma = model(x_jsma)
            preds["JSMA"].append(out_jsma.max(1)[1].cpu().numpy())
            probs["JSMA"].append(F.softmax(out_jsma, dim=1)[:, 1].cpu().numpy())

        # New Attack & Ablations
        x_new_list = []
        x_no_lap_list = []
        x_local_lap_list = []
        
        for i in range(batch_size):
            target = (1 - y[i]).item()
            baseline = torch.zeros_like(x[i])
            
            # 1. Full NewAttack (Integrated Hessian Laplacian)
            x_new_list.append(generate_attack(model, x[i], baseline, target, eps=epsilon, lam=0.1, use_integrated=True))
            
            # 2. Ablation: No Laplacian Penalty
            x_no_lap_list.append(generate_attack(model, x[i], baseline, target, eps=epsilon, lam=0.0))
            
            # 3. Ablation: Local Hessian Laplacian
            x_local_lap_list.append(generate_attack(model, x[i], baseline, target, eps=epsilon, lam=0.1, use_integrated=False))
            
        x_new = torch.stack(x_new_list)
        x_no_lap = torch.stack(x_no_lap_list)
        x_local_lap = torch.stack(x_local_lap_list)
        
        with torch.no_grad():
            out_new = model(x_new)
            preds["NewAttack"].append(out_new.max(1)[1].cpu().numpy())
            probs["NewAttack"].append(F.softmax(out_new, dim=1)[:, 1].cpu().numpy())

            out_no_lap = model(x_no_lap)
            preds["NewAttack_NoLap"].append(out_no_lap.max(1)[1].cpu().numpy())
            probs["NewAttack_NoLap"].append(F.softmax(out_no_lap, dim=1)[:, 1].cpu().numpy())

            out_local_lap = model(x_local_lap)
            preds["NewAttack_LocalLap"].append(out_local_lap.max(1)[1].cpu().numpy())
            probs["NewAttack_LocalLap"].append(F.softmax(out_local_lap, dim=1)[:, 1].cpu().numpy())

        # Compute Distortions for all attacks in this batch
        attack_examples = {
            "FGSM": x_fgsm,
            "PGD": x_pgd,
            "JSMA": x_jsma,
            "NewAttack": x_new,
            "NewAttack_NoLap": x_no_lap,
            "NewAttack_LocalLap": x_local_lap,
        }

        for name, x_adv in attack_examples.items():
            d = compute_distortions(x, x_adv)
            for norm in ["L0", "L2", "Linf"]:
                distortions[name][norm].append(d[norm])

        processed_samples += batch_size
        if checkpoint_path:
            rng_state = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
            atomic_torch_save(
                {
                    "version": 1,
                    "epsilon": epsilon,
                    "total_samples": total_samples,
                    "processed_samples": processed_samples,
                    "all_labels": all_labels,
                    "preds": preds,
                    "probs": probs,
                    "all_logits": all_logits,
                    "distortions": distortions,
                    "rng_state": rng_state,
                },
                checkpoint_path,
            )
            print(
                f"      Robustness checkpoint: {processed_samples}/{total_samples}",
                end="\r",
                flush=True,
            )

    if checkpoint_path:
        print()

    all_labels = np.concatenate(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    attacks_to_eval = ["Clean", "FGSM", "PGD", "JSMA", "NewAttack", "NewAttack_NoLap", "NewAttack_LocalLap"]

    # Pre-calculate clean correctness mask for ASR
    clean_preds_all = np.concatenate(preds["Clean"])
    correct_mask = (clean_preds_all == all_labels)
    num_correct = np.sum(correct_mask)

    metrics = {}
    for attack in attacks_to_eval:
        attack_preds = np.concatenate(preds[attack])
        attack_probs = np.concatenate(probs[attack])
        
        # Adjust labels and mask for truncated attacks
        current_labels = all_labels[:len(attack_preds)]
        current_correct_mask = correct_mask[:len(attack_preds)]
        current_num_correct = np.sum(current_correct_mask)
        
        acc = np.mean(attack_preds == current_labels)
        prec = precision_score(current_labels, attack_preds, zero_division=0)
        rec = recall_score(current_labels, attack_preds, zero_division=0)
        f1 = f1_score(current_labels, attack_preds, zero_division=0)
        try:
            auc = roc_auc_score(current_labels, attack_probs)
        except:
            auc = 0.5

        prefix = f"{attack} " if attack != "Clean" else ""
        metrics[f"{prefix}Acc"] = acc
        metrics[f"{prefix}Precision"] = prec
        metrics[f"{prefix}Recall"] = rec
        metrics[f"{prefix}F1"] = f1
        metrics[f"{prefix}AUC"] = auc

        # Calculate Attack Success Rate (ASR)
        if attack != "Clean":
            if current_num_correct > 0:
                # Of those originally correct, how many are now wrong?
                successful_attacks = np.sum(attack_preds[current_correct_mask] != current_labels[current_correct_mask])
                asr = successful_attacks / current_num_correct
            else:
                asr = 0.0
            metrics[f"{prefix}ASR"] = asr

    metrics["ECE"] = compute_ece(all_logits, torch.tensor(all_labels))
    
    # Add stealth metrics to the dictionary
    for name in distortions:
        for norm in ["L0", "L2", "Linf"]:
            vals = torch.cat(distortions[name][norm])
            metrics[f"{name} {norm}"] = vals.mean().item()
            
    return metrics


# == == == == == == == == == == == == == == == == == == == == ==
#
# 4. DATA LOADING
#
# == == == == == == == == == == == == == == == == == == == == ==

def load_cicids2017(data_path, sample_size=None):
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f, skipinitialspace=True, low_memory=False))
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    label_col = 'Label' if 'Label' in df.columns else ('Attack Type' if 'Attack Type' in df.columns else None)
    if not label_col:
        label_col = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
    X = df.drop(columns=[label_col])
    y = df[label_col]
    cols_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Unnamed: 0']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_binary = y.apply(lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL TRAFFIC'] else 1)
    if sample_size and sample_size < len(X):
        from sklearn.model_selection import train_test_split
        X, _, y_binary, _ = train_test_split(X, y_binary, train_size=sample_size, stratify=y_binary, random_state=42)
    return X.values.astype(np.float32), y_binary.values.astype(np.int64)


def load_nslkdd(data_path, sample_size=None):
    train_file = os.path.join(data_path, "KDDTrain+.txt")
    if not os.path.exists(train_file):
        train_file = os.path.join(data_path, "nsl-kdd", "KDDTrain+.txt")
    df = pd.read_csv(train_file, header=None)
    if df.shape[1] == 43: df.drop(columns=[42], inplace=True)
    X = df.drop(columns=[41])
    y = df[41]
    y_binary = y.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
    for col in [1, 2, 3]:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    if sample_size and sample_size < len(X):
        from sklearn.model_selection import train_test_split
        X, _, y_binary, _ = train_test_split(X, y_binary, train_size=sample_size, stratify=y_binary, random_state=42)
    return X.values.astype(np.float32), y_binary.values.astype(np.int64)


def load_rt_iot2022(sample_size=None):
    from ucimlrepo import fetch_ucirepo
    rt_iot2022 = fetch_ucirepo(id=942)
    X = rt_iot2022.data.features
    y = rt_iot2022.data.targets
    benign_classes = ['Thing_Speak', 'Wipro_bulb']
    y_binary = y['Attack_type'].apply(lambda x: 0 if str(x).strip() in benign_classes else 1)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    if sample_size and sample_size < len(X):
        from sklearn.model_selection import train_test_split
        X, _, y_binary, _ = train_test_split(X, y_binary, train_size=sample_size, stratify=y_binary, random_state=42)
    return X.values.astype(np.float32), y_binary.values.astype(np.int64)


def load_unsw_nb15(data_path, sample_size=None):
    """Load the official UNSW-NB15 train/test partitions as binary data."""
    expected_files = [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ]
    csv_files = []
    for filename in expected_files:
        matches = glob.glob(os.path.join(data_path, "**", filename), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not find {filename} under {data_path}")
        csv_files.append(matches[0])

    df = pd.concat(
        [pd.read_csv(csv_file, low_memory=False) for csv_file in csv_files],
        axis=0,
        ignore_index=True,
    )
    if "label" not in df.columns:
        raise ValueError("UNSW-NB15 data does not contain the expected 'label' column")

    # `attack_cat` is the multiclass form of the target and would leak the
    # binary label. `id` is only a row identifier, so neither is a feature.
    y_binary = pd.to_numeric(df["label"], errors="coerce")
    valid_rows = y_binary.notna()
    X = df.loc[valid_rows].drop(columns=["id", "attack_cat", "label"], errors="ignore").copy()
    y_binary = y_binary.loc[valid_rows].astype(np.int64)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if sample_size and sample_size < len(X):
        X, _, y_binary, _ = train_test_split(
            X,
            y_binary,
            train_size=sample_size,
            stratify=y_binary,
            random_state=42,
        )
    return X.values.astype(np.float32), y_binary.values.astype(np.int64)


# == == == == == == == == == == == == == == == == == == == == ==
#
# 5. MAIN PIPELINE
#
# == == == == == == == == == == == == == == == == == == == == ==

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the HIGAttack UNSW-NB15 experiment")
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="Comma-separated random seeds to run (default: 11,22,33,44,55)",
    )
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("At least one seed must be supplied with --seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed values are not allowed")

    datasets_to_run = []
    
     print("Checking CICIDS2017...")
     try:
         cicids_path = kagglehub.dataset_download("ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")
         X_cic, y_cic = load_cicids2017(cicids_path, sample_size=5000)
         datasets_to_run.append(("CICIDS2017", X_cic, y_cic))
     except Exception as e: print(f"Skip CICIDS2017: {e}")

     print("Checking NSL-KDD...")
     try:
         nsl_path = kagglehub.dataset_download("hassan06/nslkdd")
         X_nsl, y_nsl = load_nslkdd(nsl_path, sample_size=5000)
         datasets_to_run.append(("NSL-KDD", X_nsl, y_nsl))
     except Exception as e: print(f"Skip NSL-KDD: {e}")

     print("Checking RT-IoT2022...")
     try:
         X_rt, y_rt = load_rt_iot2022(sample_size=5000)
         datasets_to_run.append(("RT-IoT2022", X_rt, y_rt))
     except Exception as e: print(f"Skip RT-IoT2022: {e}")

     The simulated-data fallback is also disabled so a failed UNSW load
     cannot silently run a different experiment.
     if not datasets_to_run:
         print("Using simulated data...")
         X_sim = np.random.randn(2000, 78).astype(np.float32)
         y_sim = np.random.randint(0, 2, size=(2000,)).astype(np.int64)
         datasets_to_run.append(("Simulated", X_sim, y_sim))

    print("Checking UNSW-NB15...")
    try:
        unsw_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        X_unsw, y_unsw = load_unsw_nb15(unsw_path, sample_size=5000)
        datasets_to_run.append(("UNSW-NB15", X_unsw, y_unsw))
    except Exception as e:
        raise RuntimeError(f"Unable to load the required UNSW-NB15 dataset: {e}") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 0.5
    all_results = {}
    print(f"Using device {device}; selected seeds: {seeds}")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Grid Search Parameters
    lr_grid = [0.1, 0.01, 0.001]
    mom_grid = [1.0, 0.1, 0.01]

    for ds_name, X_raw, y_raw in datasets_to_run:
        print(f"\nProcessing {ds_name}...")
        ds_metrics_list = []

        for seed in seeds:
            print(f"\n  --- Run with Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # Split data: 70% Train, 15% Val, 15% Test
            X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, random_state=seed, stratify=y_raw)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train)), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val)), batch_size=64, shuffle=False)
            # A batch size of one permits an atomic resume checkpoint after
            # every expensive Integrated-Hessian test record.
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test)), batch_size=1, shuffle=False)

            best_val_acc = -1
            best_model_params = None

            print("    Performing grid search for best hyperparameters...")
            for lr in lr_grid:
                for mom in mom_grid:
                    model_path = os.path.join(model_dir, f"{ds_name}_seed{seed}_lr{lr}_mom{mom}.pth")
                    model = SecurityMLP(input_dim=X_raw.shape[1]).to(device)
                    
                    if os.path.exists(model_path):
                        load_model(model, model_path)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
                        for epoch in range(20):
                            train_baseline(model, train_loader, optimizer, device)
                        save_model(model, model_path)
                    
                    val_metrics = evaluate_clean(model, val_loader, device)
                    if val_metrics['Acc'] > best_val_acc:
                        best_val_acc = val_metrics['Acc']
                        best_model_params = (lr, mom)
            
            print(f"    Best params found: lr={best_model_params[0]}, mom={best_model_params[1]} (Val Acc: {best_val_acc:.4f})")
            
            # Load the best model to evaluate on test set
            best_lr, best_mom = best_model_params
            best_model_path = os.path.join(model_dir, f"{ds_name}_seed{seed}_lr{best_lr}_mom{best_mom}.pth")
            best_model = SecurityMLP(input_dim=X_raw.shape[1]).to(device)
            load_model(best_model, best_model_path)

            print("    Evaluating robustness on the entire test set...")
            evaluation_checkpoint = os.path.join(
                checkpoint_dir,
                f"{ds_name}_seed{seed}_robustness.pt",
            )
            metrics = evaluate_robustness(
                best_model,
                test_loader,
                eps,
                device,
                checkpoint_path=evaluation_checkpoint,
            )
            ds_metrics_list.append(metrics)

        # Aggregate Results for this dataset
        aggregated = {}
        for key in ds_metrics_list[0].keys():
            vals = [m[key] for m in ds_metrics_list]
            aggregated[f"{key}_mean"] = float(np.mean(vals))
            aggregated[f"{key}_std"] = float(np.std(vals))

        all_results[ds_name] = {
            "seeds": seeds,
            "aggregated": aggregated,
            "raw": ds_metrics_list
        }

        print(f"\nFinal Results for {ds_name} (Mean ± Std over {len(seeds)} seeds):")
        sorted_keys = sorted(ds_metrics_list[0].keys())
        for k in sorted_keys:
            mean = aggregated[f"{k}_mean"]
            std = aggregated[f"{k}_std"]
            print(f"  {k:<20}: {mean:.4f} ± {std:.4f}")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = "-".join(str(seed) for seed in seeds)
    json_path = os.path.join(output_dir, f"results_seeds_{seed_tag}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDone. Results saved to {json_path}")
