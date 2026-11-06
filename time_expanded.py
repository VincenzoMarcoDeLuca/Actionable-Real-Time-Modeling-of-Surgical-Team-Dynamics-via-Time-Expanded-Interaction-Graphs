import os
import re
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch
import utils
from utils import sessions, root_path, segment,label_order

def decode_role(x, feature_dim=88, num_phases=4):
    role_list = [
        "anest", "circulator", "assistant_surgeon", "head_surgeon",
        "mps", "nurse", "student", "unrelated_person", "unknown"
    ]
    """
    Decodifica il ruolo da un vettore feature one-hot.

    Args:
        x: feature vector (torch tensor or numpy)
        feature_dim: dimensione feature paralinguistiche
        num_phases: numero di fasi
        role_list: lista ruoli

    Returns:
        role string
    """
    role_start = feature_dim + num_phases
    role_end = role_start + len(role_list)
    role_vec = x[role_start:role_end]
    # sicurezza: converte
    role_vec = torch.tensor(role_vec)
    role_idx = torch.argmax(role_vec).item()

    return role_list[role_idx]

import torch

def decode_phase(x, feature_dim=88, phase_to_idx={0:"00_01",1:"02_04",2:"05_07",3:"08"}):
    """
    Decodifica la phase da feature vector.

    Returns:
        phase string (e.g., "00_01", "02_04", "05_07", "08")
    """
    idx_to_phase = {v: k for k, v in phase_to_idx.items()}

    phase_start = feature_dim
    phase_end = feature_dim + len(phase_to_idx)

    phase_vec = torch.tensor(x[phase_start:phase_end])

    phase_idx = torch.argmax(phase_vec).item()

    return idx_to_phase[phase_idx]

def decode_time(x):
    return x.timestep


temporal_encoding = "sinusoidal"

# --------------------------- 1. CONFIGURAZIONE ---------------------------


# 🔹 LOAD MAPPING SPEAKER → ROLE ONE-HOT
with open("speaker_to_role_one_hot.json", "r") as f:
    speaker_to_role_oh = json.load(f)

# 🔹 POSSIBILI RUOLI
role_list = [
    "anest", "circulator", "assistant_surgeon", "head_surgeon",
    "mps", "nurse", "student", "unrelated_person", "unknown"
]

num_roles = len(role_list)



# --------------------------- 2. PHASE ONE-HOT ---------------------------
all_phases = sorted({phase for sess in sessions.values() for phase in sess.keys()})
phase_to_idx = {p: i for i, p in enumerate(all_phases)}
num_phases = len(all_phases)

# --------------------------- 3. TROVA FEATURE DIMENSION ---------------------------
feature_dim = None
for session, phases in sessions.items():
    for phase, subs in phases.items():
        for sub in subs:
            base_feat = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                     "output", sub, "segments", segment, "features_paralinguistic")
            if not os.path.exists(base_feat): continue
            for s in os.listdir(base_feat):
                sp_path = os.path.join(base_feat, s)
                if os.path.isdir(sp_path):
                    files = [f for f in os.listdir(sp_path) if f.endswith(".csv")]
                    if files:
                        df = pd.read_csv(os.path.join(sp_path, files[0]))
                        feature_dim = df.select_dtypes(include=[np.number]).values.flatten().shape[0]
                        break
            if feature_dim is not None: break
        if feature_dim is not None: break
    if feature_dim is not None: break

if feature_dim is None:
    raise RuntimeError("⚠️ Feature dimension not found")

time_dim = 8
feature_dim_total = feature_dim + num_phases + num_roles + time_dim


# --------------------------- 4. TIME-EXPANDED GRAPH ---------------------------
def build_time_expanded_graph(graph_sequence, num_speakers, temporal_encoding="standard", pe_dim=8):
    """
    Costruisce il Time-Expanded Graph con edge intra-timestep e temporali,
    aggiunge la feature temporale e salva un tensore `timestep` per ogni nodo.

    Args:
        graph_sequence: lista di oggetti Data (PyG) per ciascun timestep
        num_speakers: numero massimo di speaker nel dataset
        temporal_encoding: "standard" per [0,1], "sinusoidal" per sin/cos
        pe_dim: dimensione del positional encoding sinusoidale
    Returns:
        Data (PyG) con campi: x, edge_index, timestep
    """

    def positional_encoding(t_norm, d=8):
        """Positional encoding sinusoidale"""
        pe = []
        for i in range(d):
            freq = 1 / (10000 ** (2 * i / d))
            if i % 2 == 0:
                pe.append(np.sin(2 * np.pi * t_norm * freq))
            else:
                pe.append(np.cos(2 * np.pi * t_norm * freq))
        return np.array(pe, dtype=np.float32)

    all_x = []
    edge_list = []
    node_to_timestep = []

    T = len(graph_sequence)

    for t_idx, g in enumerate(graph_sequence):
        offset = t_idx*g.x.shape[0]  # accumula nodi già aggiunti

        # 🔹 feature temporale
        if temporal_encoding == "standard":
            t_feat = torch.full((g.x.size(0), 1), t_idx / max(T - 1, 1))
        elif temporal_encoding == "sinusoidal":
            t_norm = t_idx / max(T - 1, 1)
            pe = positional_encoding(t_norm, d=pe_dim)
            t_feat = torch.tensor(pe, dtype=torch.float32).unsqueeze(0).repeat(g.x.size(0), 1)
        else:
            raise ValueError(f"Temporal encoding {temporal_encoding} not recognized")

        # 🔹 concat timestep alla feature
        x_with_time = torch.cat([g.x, t_feat], dim=1)
        all_x.append(x_with_time)

        # 🔹 edge intra-timestep
        for i, j in g.edge_index.t().tolist():
            edge_list.append([i + offset, j + offset])

        # 🔹 salva timestep per ciascun nodo
        node_to_timestep.extend([t_idx] * g.x.size(0))

    # 🔹 edge temporali tra timestep consecutivi per ciascun speaker
    for t_idx in range(1, T):
        prev_offset = sum([graph_sequence[t].x.size(0) for t in range(t_idx)])
        curr_offset = sum([graph_sequence[t].x.size(0) for t in range(t_idx + 1)])
        for sp in range(num_speakers):
            # verifica che esistano nodi per questo speaker in entrambi i timestep
            prev_idx = prev_offset - graph_sequence[t_idx - 1].x.size(0) + sp
            curr_idx = curr_offset - graph_sequence[t_idx].x.size(0) + sp
            if prev_idx < prev_offset and curr_idx < curr_offset:
                edge_list.append([prev_idx, curr_idx])
                edge_list.append([curr_idx, prev_idx])

    x = torch.cat(all_x, dim=0)
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    timestep_tensor = torch.tensor(node_to_timestep, dtype=torch.long)
    T = len(graph_sequence)  # 🔥 QUI

    # 🔹 salva tutto nel Data PyG
    return Data(
        x=x,
        edge_index=edge_index,
        timestep=timestep_tensor,   # 🔥 AGGIUNGI QUESTO
        session=torch.tensor([int(session[:3])]),
        phase=torch.tensor([phase_to_idx[phase]]),
        sub = torch.tensor([int(sub)])
    )

# --------------------------- 5. COSTRUZIONE DATASET ---------------------------
dataset = []
labels = []

for session, phases in sessions.items():

    max_speaker_id = -1
    for phase, subs in phases.items():
        for sub in subs:
            base_feat = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                     "output", sub, "segments", segment, "features_paralinguistic")
            if not os.path.exists(base_feat): continue
            for s in os.listdir(base_feat):
                m = re.match(r"SPEAKER_(\d+)", s)
                if m:
                    max_speaker_id = max(max_speaker_id, int(m.group(1)))
    num_speakers = max_speaker_id + 1
    if num_speakers == 0: continue

    for phase, subs in phases.items():
        for sub in subs:
            base_feat = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                     "output", sub, "segments", segment, "features_paralinguistic")
            diar_base = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                     "output", sub, "segments", segment, "diarizzazioni_diarizzati")
            stats_path = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                      "output", sub, "labels", "stats.json")

            if not os.path.exists(base_feat) or not os.path.exists(stats_path): continue

            phase_oh = np.zeros(num_phases)
            phase_oh[phase_to_idx[phase]] = 1

            timesteps = set()
            for s in os.listdir(base_feat):
                sp_path = os.path.join(base_feat, s)
                if os.path.isdir(sp_path):
                    timesteps.update([f for f in os.listdir(sp_path) if f.endswith(".csv")])
            if len(timesteps) == 0: continue
            timesteps = sorted(timesteps)

            with open(stats_path, "r") as f:
                stats = json.load(f)
            try:
                Y = np.array([stats[m]["Score_mean"] for m in label_order])
                Y = np.clip(Y, 0, 6)#.astype(int)
                Y = torch.tensor(Y)#, dtype=torch.long)
            except:
                continue

            graph_sequence = []

            for t in timesteps:
                X_t = []
                active_speakers = []

                for sp_id in range(num_speakers):
                    sp_name = f"SPEAKER_{sp_id:02d}"

                    # 🔹 ROLE ONE-HOT
                    role_oh = speaker_to_role_oh.get(session, {}).get(sp_name, [0]*num_roles)

                    # 🔹 feature
                    file_path = os.path.join(base_feat, sp_name, t)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        arr = df.select_dtypes(include=[np.number]).values.flatten()
                        arr = np.nan_to_num(arr)
                    else:
                        arr = np.zeros(feature_dim)

                    arr = np.concatenate([arr, phase_oh, role_oh])
                    X_t.append(arr)

                    # 🔹 active speakers
                    diar_path = os.path.join(diar_base, sp_name, t.replace(".csv",".txt"))
                    if os.path.exists(diar_path) and os.path.getsize(diar_path) > 0:
                        active_speakers.append(sp_id)

                X_t = torch.tensor(np.stack(X_t), dtype=torch.float)

                A = np.zeros((num_speakers, num_speakers))
                for i in active_speakers:
                    for j in range(num_speakers):
                        if i != j:
                            A[i, j] = 1
                edge_index, _ = dense_to_sparse(torch.tensor(A, dtype=torch.float))
                g = Data(
                    x=X_t,
                    edge_index=edge_index,
                    session=torch.tensor([int(session[:3])]),
                    phase=torch.tensor([phase_to_idx[phase]]),
                    sub = torch.tensor([int(sub)])
                )
                graph_sequence.append(g)

            print("Len: graph_sequence: ", len(graph_sequence))
            if len(graph_sequence) == 0: continue

            teg = build_time_expanded_graph(graph_sequence, num_speakers, temporal_encoding = "sinusoidal")
            dataset.append(teg)
            labels.append(Y)

# --------------------------- 6. SAVE ---------------------------
save_dir = os.path.join(root_path, "processed_dataset_time_expanded")
os.makedirs(save_dir, exist_ok=True)

torch.save(dataset, os.path.join(save_dir, "graph_sequences_time_expanded.pt"))
torch.save(labels, os.path.join(save_dir, "labels_time_expanded.pt"))

# --------------------------- 5b. SOLO SESSIONE/FASE/SOTTOFASE/LABEL ---------------------------
summary_info = []

for session, phases in sessions.items():
    for phase, subs in phases.items():
        for sub in subs:
            stats_path = os.path.join(root_path, session, "NEW_SPLIT", phase,
                                      "output", sub, "labels", "stats.json")
            if not os.path.exists(stats_path):
                continue

            with open(stats_path, "r") as f:
                stats = json.load(f)
            try:
                Y = np.array([stats[m]["Score_mean"] for m in label_order])
                Y = np.clip(Y, 0, 7)
                summary_info.append({
                    "session": session,
                    "phase": phase,
                    "subphase": sub,
                    "labels": Y.tolist()
                })
            except:
                continue
# Salvataggio opzionale in CSV
summary_df = pd.DataFrame(summary_info)
summary_df.to_csv(os.path.join(root_path, "session_phase_subphase_labels.csv"), index=False)
print(f"✅ Saved summary for {len(summary_info)} subphases")
