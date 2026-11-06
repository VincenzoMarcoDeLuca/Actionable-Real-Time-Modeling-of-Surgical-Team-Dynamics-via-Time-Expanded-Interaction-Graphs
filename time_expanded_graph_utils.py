import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GINConv
import torch.nn.functional as F

# ---------------------------
# 6. TGNN MODEL STABILIZZATO
# ---------------------------
# ---------------------------
# 6. TGNN MODEL STABILIZZATO CON FUNZIONE ENCODE
# ---------------------------
class TGNN_TimeExpanded(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, paralinguistic_dim = 88, gnn_type="GCN", dropout_prob=0.2, noise_std=0.0, heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.gnn_type = gnn_type
        self.paralinguistic_dim = paralinguistic_dim

        if gnn_type == "GCN":
            self.gnn1 = GCNConv(in_feats, hidden_size)
            self.gnn2 = GCNConv(hidden_size, hidden_size)
            norm_dim = hidden_size
        elif gnn_type == "GAT":
            self.gnn1 = GATConv(in_feats, hidden_size, heads=heads)
            self.gnn2 = GATConv(hidden_size*heads, hidden_size, heads=heads)
            norm_dim = hidden_size*heads
        elif gnn_type == "GIN":
            nn1 = nn.Sequential(nn.Linear(in_feats, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size))
            nn2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size))
            self.gnn1 = GINConv(nn1)
            self.gnn2 = GINConv(nn2)
            norm_dim = hidden_size
        else:
            raise ValueError("gnn_type must be 'GCN', 'GAT', or 'GIN'")

        self.norm1 = nn.LayerNorm(norm_dim)
        self.norm2 = nn.LayerNorm(norm_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(norm_dim, num_classes)

    def forward(self, graphs, device=None):
        out_list = []
        for g in graphs:
            x, edge_index = g.x.to(device), g.edge_index.to(device)

            if self.training and self.noise_std > 0:
                # 🔹 Aggiungi noise SOLO alle feature continue (paralinguistic)
                x[:, :self.paralinguistic_dim] += torch.randn_like(x[:, :self.paralinguistic_dim]) * self.noise_std

                # ✅ NON toccare phase/role/time one-hot
                # x[:, start_phase:start_role], x[:, start_role:start_time], x[:, start_time:]

            h1 = F.relu(self.gnn1(x, edge_index))
            h1 = self.norm1(h1)
            h2 = F.relu(self.gnn2(h1, edge_index) + h1)
            h2 = self.norm2(h2)
            h2 = self.dropout(h2)

            h_pool = global_mean_pool(h2, torch.zeros(h2.size(0), dtype=torch.long, device=h2.device))
            h_pool = h_pool.squeeze(0)
            out = self.fc(h_pool)
            out_list.append(out)
        return torch.stack(out_list)

    def encode(self, graphs, device=None):
        """
        Restituisce embedding della sequenza senza passare per il classificatore
        """
        embedding_list = []
        for g in graphs:
            try:
                x, edge_index = g.x.to(device), g.edge_index.to(device)
            except:
                import pdb; pdb.set_trace()
            h1 = F.relu(self.gnn1(x, edge_index))
            h1 = self.norm1(h1)
            h2 = F.relu(self.gnn2(h1, edge_index) + h1)
            h2 = self.norm2(h2)
            # pooling globale
            h_pool = global_mean_pool(h2, torch.zeros(h2.size(0), dtype=torch.long, device=h2.device))
            h_pool = h_pool.squeeze(0)
            embedding_list.append(h_pool)
        # embedding medio della sequenza
        return torch.stack(embedding_list).mean(dim=0)