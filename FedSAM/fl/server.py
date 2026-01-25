import torch

def server_update(global_model, clients_state_list, data_size_list):
    global_state = global_model.state_dict()
    new_state = {name: torch.zeros_like(p) for name, p in global_state.items()}

    data_size_sum = sum(data_size_list)
    client_size = len(clients_state_list)

    for name in new_state.keys():
        new_state[name] = sum( (data_size_list[i] / data_size_sum) * clients_state_list[i][name] for i in range(client_size) )

    global_model.load_state_dict(new_state)

    return global_model