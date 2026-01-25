# FedSAM on CIFAR-10 and MNIST with MobileNet (PyTorch)

This repository provides a PyTorch implementation of FedSAM (Generalized Federated Learning via Sharpness-Aware Minimization) using MobileNet on CIFAR-10 and MNIST. 

FedSAM improves robustness under Non-IID data by training each client with a sharpness-aware objective (SAM), then aggregating client models on the server.

⸻

## Project Overview

This simulation covers:
	•	Data Partitioning: IID and Non-IID splits (Dirichlet distribution).
	•	Client-side FedSAM: local training uses two-step SAM update (perturb → descent).
	•	Server-side Aggregation: standard weighted FedAvg aggregation of client models (Same as FedAVG aggregation).
	•	BatchNorm Handling (stability):
	•	SAM training typically disables BN running-stat updates during the first forward pass.
	•	Server aggregation can average BN buffers (running_mean/var) with standard averaging (no special correction).

⸻

## Recommended Folder Structure

Your `main.py` imports modules like `data.cifar10`, `models.mobilenet,` etc.
Organize files like this to run without modifying imports:
```
├── main.py
├── data/
│   ├── __init__.py
│   ├── cifar10.py
│   ├── mnist.py
│   └── partition.py
├── fl/
│   ├── __init__.py
│   ├── fedsam.py
│   └── server.py
├── models/
│   ├── __init__.py
│   └── mobilenet.py
└── utils/
    ├── __init__.py
    ├── device.py
    ├── eval.py
    ├── parser.py
    └── seed.py
```

## Requirements

- Python 3.9+ recommended
- PyTorch + torchvision
- numpy

Run with default settings:
```bash
python main.py
```
Example1: Non-IID
```bash
python main.py --partition niid
```
Example2: Non-IID with control dyn-alpha
```bash
python main.py --partition niid --alpha 0.5 --min-size 10
```
Example3: Change the number and ratio of participating clients + local epoch
```bash
python main.py --num-clients 100 --client-frac 0.2 --local-epochs 5
```

## Device Selection

The code supports:
```
	•	--device auto (default): selects CUDA if available, else MPS (Apple Silicon), else CPU
	•	--device cuda
	•	--device mps
	•	--device cpu
```

Example:
```bash
python main.py --device auto
```

## CLI Arguments

Key arguments (from utils/parser.py):
```
	•	Reproducibility / compute
	  •	--seed (default: 845)
	  •	--device in {auto,cpu,cuda,mps}

	•	Training method
	  •	--sam-rho (FedSAM rho, default 0.05)

	•	Dataset
	  •   --data-set (default cifar10, choices=[cifar10, mnist])
	  •	--data-root (default ./data)
	  •	--augment / --no-augment
	  •	--normalize / --no-normalize
	  •	--test-batch-size (default 128)

	•	Federated learning config
	  •	--num-clients (default 10)
	  •	--client-frac fraction of clients sampled per round (default 0.25)
	  •	--local-epochs (default 1)
	  •	--batch-size (default 100)
	  •	--lr learning rate (default 1e-2)
	  •	--rounds communication rounds (default 10)

	•	Data partitioning
	  •	--partition in {iid,niid}
	  •	--alpha: Dirichlet concentration parameter controlling Non-IID severity.
		      ├── α = 0.1 ~ 0.3: highly skewed label distribution (strong Non-IID)
		  	  ├──	α = 0.5: moderate Non-IID (default)
		  	  └──	α = 0.8 ~ 1.0: closer to IID
	  •	--min-size minimum samples per client in non-IID (default 10)
	  •	--print-labels / --no-print-labels

	•	Learning rate Scheduler (ReduceOnPlateau)
	  •	--lr-factor (learning rate * factor, default 0.5)
	  •	--lr-patience (default 5)
	  •	--min-lr (deafult 1e-6)
	  •	--lr-threshold (default 1e-4)
	  •	--lr-cooldown (default 0)
```

## FedSAM Implementation Notes

### 1) Client-side FedSAM Update (fl/fedsam.py)

FedSAM applies Sharpness-Aware Minimization (SAM) locally on each client.

**SAM objective (per client):**
$$\min_{\theta}\; \max_{\|\epsilon\|\le\rho}\; \mathcal{L}_k(\theta + \epsilon)$$

A standard two-step SAM update is:
	1.	Compute gradient at current weights:
$
\nabla g = \nabla_{\theta} \mathcal{L}_k(\theta)
$
	2.	Perturb weights toward the gradient direction:
$
\epsilon^* = \rho \frac{\nabla_{\theta} \mathcal{L}_k(\theta)}{\|\nabla_{\theta} \mathcal{L}_k(\theta)\|_2}
$
	3.	Compute gradient at perturbed weights and do the descent step:
$
\quad \theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}_k(\theta + \epsilon^*)
$

BatchNorm note (recommended): During the “perturb forward/backward”, many SAM implementations disable BN running-stat updates for stability.

Optimizer: typically SGD with weight decay.

⸻

## 2) Server-side Aggregation (fl/server.py)

**The server aggregates client models with standard weighted averaging (FedAvg)**:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^t$$
	• (n_k: number of samples at client k)

BatchNorm buffers (running_mean/var, num_batches_tracked): 
	• In practice, you can aggregate them using the same weighted averaging (or optionally keep them local depending on your setting).

## Expected Output

Each round prints evaluation results like:
```bash
=== Evaluate global model 1 Round ===
[01] acc=XX.XX%, loss=Y.YYYYYY
```

With data_set="cifar10", num_clients=100, client_frac=0.25, local_epochs=5, batch_size=50, lr=1e-2, rounds=200, partition="niid", alpha=0.4, lr_patience=10, min_lr=1e-5:
<br>79 Round ACC=60.91%, loss=1.128906
<br>85 Round ACC=62.36%, loss=1.098357
<br>93 Round ACC=65.28%, loss=1.024822
<br>109 Round ACC=69.61%, loss=0.944319
<br>132 Round ACC=73.68%, loss=0.800382
<br>151 Round ACC=76.91%, loss=0.739485
<br>166 Round ACC=78.16%, loss=0.676211
<br>183 Round ACC=78.83%, loss=0.631958
<br>191 Round ACC=81.23%, loss=0.580031
<br>200 Round ACC=81.77%, loss=0.594970
