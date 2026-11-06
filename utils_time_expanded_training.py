from torch.utils.data import Dataset
import torch
# ---------------------------
# 4. CUSTOM DATASET
# ---------------------------
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# 6️⃣ Dataset personalizzato (graph, label)
# ---------------------------
class GraphWithLabelDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

