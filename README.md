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
$g = \nabla_{\theta} \mathcal{L}_k(\theta)$
	2.	Perturb weights toward the gradient direction:
$\epsilon = \rho \cdot \frac{g}{\|g\| + \varepsilon},\quad \theta^{+}=\theta+\epsilon$
	3.	Compute gradient at perturbed weights and do the descent step:
$\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}_k(\theta^{+})$

BatchNorm note (recommended):
	•	During the “perturb forward/backward”, many SAM implementations disable BN running-stat updates for stability.

Optimizer: typically SGD with weight decay.

⸻

## 2) Server-side Aggregation (fl/server.py)

**The server aggregates client models with standard weighted averaging (FedAvg)**:

$\theta^{t} = \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j\in\mathcal{S}_t} n_j}\;\theta_k^{t}$
	•	\mathcal{S}_t: participating clients in round t
	•	n_k: number of samples at client k

BatchNorm buffers (running_mean/var, num_batches_tracked):
	•	In practice, you can aggregate them using the same weighted averaging (or optionally keep them local depending on your setting).
  이거야
## FedDyn Implementation Notes

### 1) Client-side Update (fl/feddyn.py)

Each client minimizes a dynamically regularized objective to reduce client drift from the global optimum.

**Local objective (per client):**

$$𝝷_k^t = L_{total}(𝝷) - {\langle g_k^{t-1}, 𝝷\rangle} + \frac{\alpha}{2} * |\theta-\theta^{t-1}\|^2$$

- $L_{\text{task}}$: standard cross-entropy loss on local batch $b$.
- $-\langle 𝝷_k^{t}, \theta \rangle$: linear correction term using the client-specific state $h_k^t$.
- $\frac{\alpha}{2}\|\theta-\theta^{t}\|^2$: proximal term keeping the local model close to the global model $\theta^t$.

**Optimizer:** SGD with `momentum=0.9`, `weight_decay=5e-4`.

**Client state update (after local training):**

$$
g_k^{t} = g_k^{t-1} - \alpha(\theta_k^{t}-\theta^{t-1})
$$

where $\theta_k^{t+1}$ is the client model after local training and $\theta^{t}$ is the global model received at the start of round $t$.

⸻

2) Server-side Aggregation (fl/server.py)

The server maintains a global correction state $h$ and updates the global model using a corrected averaging scheme.

(a) Server state $h$ update:
$$h^{t} = h^{t-1} - \alpha \cdot \frac{1}{m}\sum_{k\in P_i}(\theta_k^{t}-\theta^{t-1})$$<br>
	•	$m$: Number of all clients<br>
	•	The server state $$h$$ accumulates the average drift $$(\theta_k^{t}-\theta^{t-1})$$ across every participating clients.

(b) Global model update
For learnable parameters (weights/bias):

$$\\overline{\theta^{t}} = \frac{1}{P}\sum_{k\in P_i}\theta_k^{t}$$

$$\theta^t = \\overline{\theta^{t}} - \frac{1}{\alpha}h^{t}$$

For BatchNorm buffers (e.g., running_mean, running_var, num_batches_tracked):

$$\theta^{t} = \\overline{\theta^{t}}$$

BatchNorm buffers are aggregated by simple averaging (no FedDyn correction).

## Expected Output

Each round prints evaluation results like:
```bash
=== Evaluate global model 1 Round ===
[01] acc=XX.XX%, loss=Y.YYYYYY
```

With data_set="cifar10", num_clients=100, client_frac=0.25, local_epochs=5, batch_size=50, lr=1e-2, rounds=200, partition="niid", alpha=0.4, lr_patience=10, min_lr=1e-5:
<br>83 Round ACC=60.65%, loss=1.122256
<br>96 Round ACC=63.43%, loss=1.039685
<br>106 Round ACC=65.97%, loss=1.004173
<br>117 Round ACC=67.29%, loss=0.951618
<br>128 Round ACC=68.31%, loss=0.939058
<br>134 Round ACC=69.24%, loss=0.933948
<br>142 Round ACC=69.70%, loss=0.896229
<br>144 Round ACC=70.21%, loss=0.907815
<br>145 Round ACC=70.63%, loss=0.875592
<br>151 Round ACC=71.42%, loss=0.848190
<br>152 Round ACC=72.03%, loss=0.853616
<br>159 Round ACC=72.41%, loss=0.816083
<br>163 Round ACC=73.27%, loss=0.789843
<br>167 Round ACC=73.91%, loss=0.774350
<br>189 Round ACC=74.31%, loss=0.742489
<br>200 Round ACC=75.38%, loss=0.723625



