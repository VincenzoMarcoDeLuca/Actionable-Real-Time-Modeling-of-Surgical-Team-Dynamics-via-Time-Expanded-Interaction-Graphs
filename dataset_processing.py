import numpy as np
import torch
label_order = ["Communication","Coordination","Cooperation","Leadership","Monitoring"]

mapping = {
        0.0: 0,
        0.5: 0,
        1.0: 0,
        1.5: 0,
        2.0: 0,
        2.5: 1,
        3.0: 2,
        3.5: 3,
        4.0: 3,
        4.5: 4,
        5.0: 4,
        5.5: 4,
        6.0: 4
    }

def feature_normalization(dataset):
    paraling_dim, phase_dim, role_dim = 88, 4, 9
    start_phase, start_role, start_time = paraling_dim, paraling_dim + phase_dim, paraling_dim + phase_dim + role_dim

    all_x = torch.cat([g.x[:, :start_time] for g in dataset], dim=0)
    mean, std = all_x[:, :paraling_dim].mean(0), all_x[:, :paraling_dim].std(0) + 1e-5

    for g in dataset:
        g.x = g.x.clone()
        g.x[:, :start_time] = g.x[:, :start_time]  # paraling + phase + role
        g.x[:, :paraling_dim] = (g.x[:, :paraling_dim] - mean) / std
    return mean, std

def discretize_fixed_5classes(values):
    print("\n📊 RAW VALUE DISTRIBUTION")

    unique, counts = np.unique(values, return_counts=True)

    for u, c in zip(unique, counts):
        print(f"Value {u}: {c} ({c/len(values):.2%})")


    return np.array([mapping[v] for v in values])

def bin_labels(labels, social_dimension):
    coord_idx = label_order.index(social_dimension)

    # Usa ESATTAMENTE la stessa discretizzazione
    # definita precedentemente
    coord_raw_values = np.array([
        y[coord_idx].item()
        for y in labels
    ])
    labels_coord = discretize_fixed_5classes(
        coord_raw_values
    )
    return labels_coord

def filter_out(labels, social_dimension):
    coord_idx = label_order.index(social_dimension)
    return np.array([y[coord_idx].item() for y in labels])