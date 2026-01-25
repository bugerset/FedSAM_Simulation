import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader

def disable_running_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            m.backup_momentum = m.momentum
            m.momentum = 0

def enable_running_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            m.momentum = m.backup_momentum

def local_train_fedsam(global_model, client_dataset, epochs, batch_size, lr, device="cpu",
                        rho=0.05, weight_decay=5e-4):
    
    local_model =  copy.deepcopy(global_model).to(device)
    local_model.train()

    loader = DataLoader(dataset=client_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_k = len(client_dataset)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            disable_running_stats(local_model)
            optimizer.zero_grad()

            pred_perturbation = local_model(x)
            loss_perturbation = criterion(pred_perturbation, y)
            loss_perturbation.backward()

            grads = [p.grad for p in local_model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
            scale = rho / (grad_norm + 1e-12)

            with torch.no_grad():
                for p in local_model.parameters():
                    if p.grad is None:
                        continue
                    epsilon = p.grad * scale
                    p.add_(epsilon)
                    p.epsilon = epsilon

            enable_running_stats(local_model)
            optimizer.zero_grad()

            pred = local_model(x)
            loss = criterion(pred, y)
            loss.backward()

            with torch.no_grad():
                for p in local_model.parameters():
                    if p.grad is None:
                        continue
                    p.sub_(p.epsilon)
                    del p.epsilon

            optimizer.step()

    client_state = {name: p.detach().cpu().clone() for name, p in local_model.state_dict().items()}

    return client_state, n_k    