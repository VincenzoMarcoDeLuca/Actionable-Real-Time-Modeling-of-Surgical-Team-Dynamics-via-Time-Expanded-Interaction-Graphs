import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from dataset_processing import bin_labels, label_order, filter_out, feature_normalization
from utils_time_expanded_training import GraphDataset, GraphWithLabelDataset
from time_expanded_graph_utils import TGNN_TimeExpanded
import yaml

num_classes = 5

with open(rf"config\te_renn\communication\gin.yaml", "r") as f:
    cfg = yaml.safe_load(f)

with open(rf"config\time_expanded_generation.yaml", "r") as f:
    cfg_time_expanded_generation = yaml.safe_load(f)

base_path = cfg_time_expanded_generation["root_dir"]
root_path = f"{base_path}\Labeling"
save_dir = f"{root_path}\\processed_dataset_time_expanded"
with open(rf"config\te_renn\communication\gin.yaml", "r") as f:
    cfg = yaml.safe_load(f)

dataset_path = f"{save_dir}\\graph_sequences_time_expanded.pt"
labels_path  = f"{save_dir}\\labels_time_expanded.pt"

dataset = torch.load(dataset_path)
labels = torch.load(labels_path)

# ---------------------------
# 3️⃣ Normalizzazione features
# ---------------------------
mean, std = feature_normalization(dataset)

# ---------------------------
# 4️⃣ Label Coordination
# ---------------------------
social_dimension = cfg["social_dimension"]

labels_coord = bin_labels(labels, social_dimension)

print(f"\n📊 {social_dimension} - LABEL FINALI")

unique_coord, counts_coord = np.unique(labels_coord,
    return_counts=True
)

for c, n in zip(unique_coord, counts_coord):
    print(
        f"Class {c}: {n} samples "
        f"({n / len(labels_coord):.2%})"
    )


# ---------------------------
# 5️⃣ Filtra classi
# ---------------------------

classes, counts = np.unique(
    labels_coord,
    return_counts=True
)

# Manteniamo solo le classi con almeno 2 campioni
valid_classes = classes[counts >= 2]

valid_indices = [
    i
    for i, lbl in enumerate(labels_coord)
    if lbl in valid_classes
]

filtered_labels = labels_coord[valid_indices]

filtered_graphs = [
    dataset[i]
    for i in valid_indices
]

print(
    f"\n✅ Dopo filtraggio: "
    f"{len(valid_classes)} classi valide "
    f"su {len(classes)} totali"
)

print(
    f"✅ Campioni filtrati: "
    f"{len(filtered_graphs)}"
)

print(
    f"✅ Classi valide: "
    f"{valid_classes.tolist()}"
)
print(f"✅ Dopo filtraggio: {len(valid_classes)} classi valide su {len(classes)} totali")
print(f"✅ Campioni filtrati: {len(filtered_graphs)}")


vals = filter_out(labels, social_dimension)

# ---------------------------

dataset_obj = GraphWithLabelDataset(filtered_graphs, filtered_labels)

# ---------------------------
# 7️⃣ Stratified split
# ---------------------------
# split_ratio = 0.75
# sss = StratifiedShuffleSplit(n_splits=1, train_size=split_ratio, random_state=cfg["random_state"]) #First trail:42
# train_idx, test_idx = next(sss.split(np.zeros(len(filtered_labels)), filtered_labels))
# train_subset = Subset(dataset_obj, train_idx)
# test_subset  = Subset(dataset_obj, test_idx)

split_path = os.path.join(
    save_dir,
    f"split_HigherOrderTransformer_{social_dimension}.pt"
)

# ---------------------------
# 7️⃣ CARICA LO SPLIT SALVATO
# ---------------------------

if not os.path.exists(split_path):
    raise FileNotFoundError(
        f"\n❌ Split salvato non trovato:\n{split_path}\n\n"
        "Devi prima eseguire lo script che crea e salva "
        "train_idx / val_idx / test_idx."
    )

print("\n📂 Caricamento split salvato...")
split_data = torch.load(
    split_path,
    map_location="cpu",
    weights_only=False
)

# Indici dello split salvato
train_idx = np.asarray(split_data["train_idx"], dtype=np.int64)
val_idx   = np.asarray(split_data["val_idx"], dtype=np.int64)
test_idx  = np.asarray(split_data["test_idx"], dtype=np.int64)

# Indici originali utilizzati per creare filtered_graphs
saved_valid_indices = np.asarray(
    split_data["valid_indices"],
    dtype=np.int64
)

current_valid_indices = np.asarray(
    valid_indices,
    dtype=np.int64
)

# ---------------------------
# Controllo fondamentale
# ---------------------------
if not np.array_equal(
    saved_valid_indices,
    current_valid_indices
):
    raise RuntimeError(
        "\n❌ ERRORE: il dataset filtrato attuale non corrisponde "
        "a quello con cui è stato creato lo split salvato.\n\n"
        "Probabilmente il dataset è cambiato oppure l'ordine dei grafi "
        "è diverso.\n"
        "Per sicurezza NON continuo il training."
    )

print("✅ Lo split salvato corrisponde al dataset attuale.")

# ---------------------------
# Controllo indici
# ---------------------------
all_indices = set(range(len(filtered_graphs)))

train_set = set(train_idx.tolist())
val_set   = set(val_idx.tolist())
test_set  = set(test_idx.tolist())

if train_set & val_set:
    raise RuntimeError("❌ Train e Validation hanno campioni in comune.")

if train_set & test_set:
    raise RuntimeError("❌ Train e Test hanno campioni in comune.")

if val_set & test_set:
    raise RuntimeError("❌ Validation e Test hanno campioni in comune.")

if (train_set | val_set | test_set) != all_indices:
    raise RuntimeError(
        "❌ Train + Validation + Test non coprono esattamente "
        "tutti i campioni filtrati."
    )

# ---------------------------
# Crea i Subset
# ---------------------------
train_subset = Subset(dataset_obj, train_idx.tolist())
val_subset   = Subset(dataset_obj, val_idx.tolist())
test_subset  = Subset(dataset_obj, test_idx.tolist())

print("\n📊 SPLIT CARICATO DA DISCO")
print(f"Train      : {len(train_subset)}")
print(f"Validation : {len(val_subset)}")
print(f"Test       : {len(test_subset)}")
print(
    f"Totale     : "
    f"{len(train_subset) + len(val_subset) + len(test_subset)}"
)





# ---------------------------
# 📊 CHECK SESSIONI UTILIZZATE
# ---------------------------

# ---------------------------
# 📊 CHECK SESSIONI UTILIZZATE
# ---------------------------

used_sessions = set()
session_counts = {}

for i in range(len(dataset_obj)):
    graph, label = dataset_obj[i]   # 🔥 FIX QUI

    sess = int(graph.session.item())

    used_sessions.add(sess)
    session_counts[sess] = session_counts.get(sess, 0) + 1

print("\n📊 SESSIONI UTILIZZATE NEL DATASET COMPLETO:")
for s in sorted(used_sessions):
    print(f"Session {s}: {session_counts[s]} graphs")

print("\n📌 Totale sessioni usate:", len(used_sessions))
print("📌 Totale grafi:", len(dataset_obj))

# ---------------------------
# 📊 CHECK DISTRIBUZIONE LABELS (GLOBAL)
# ---------------------------

unique, counts = np.unique(filtered_labels, return_counts=True)

print("\n📊 DISTRIBUZIONE CLASSE (GLOBAL - filtered_labels)")
for c, n in zip(unique, counts):
    print(f"Class {c}: {n} samples ({n/len(filtered_labels):.2%})")

print("\n📌 Summary:")
print(f"- Total samples: {len(filtered_labels)}")
print(f"- Num classes: {len(unique)}")
print(f"- Min class count: {counts.min()}")
print(f"- Max class count: {counts.max()}")
print(f"- Imbalance ratio (max/min): {counts.max() / (counts.min() + 1e-8):.2f}")

torch.save(train_subset, os.path.join(save_dir, "train_data_list_2.pt"))
torch.save(test_subset, os.path.join(save_dir, "test_data_list_2.pt"))
print(f"✅ Train size: {len(train_subset)}, Test size: {len(test_subset)}")



# ---------------------------
# 8️⃣ Collate function
# ---------------------------
def collate_fn(batch):
    graphs, labels = zip(*batch)
    return graphs, torch.stack(labels)

# ---------------------------
# 9️⃣ Loss function
# ---------------------------
def ordinal_loss(outputs, targets, class_weights_tensor, alpha=0.1):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(outputs, targets)
    probs = torch.softmax(outputs, dim=1)
    value_pred = torch.sum(probs * torch.arange(probs.size(1), device=outputs.device), dim=1)
    ord_loss = nn.MSELoss()(value_pred, targets.float())
    return ce_loss, ord_loss, ce_loss + alpha*ord_loss

# ---------------------------
# 10️⃣ Modello, optimizer, scheduler
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TGNN_TimeExpanded(
    in_feats=dataset[0].x.shape[1],
    hidden_size=cfg["hidden_size"],
    num_classes=num_classes,
    gnn_type=cfg["gnn_type"],
    dropout_prob=cfg["dropout_prob"],
    noise_std=cfg["noise_std"]
).to(device)

# Pesatura classi

train_labels = filtered_labels[train_idx]

train_classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=train_classes,
    y=train_labels
)

class_weights = np.ones(
    num_classes,
    dtype=np.float32
)

for cls, w in zip(train_classes, weights):
    class_weights[int(cls)] = w

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=50, threshold=1e-4)


# ---------------------------
# 10️⃣ DataLoader
# ---------------------------

train_loader = DataLoader(
    train_subset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_subset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_subset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

print("\n📦 DATALOADER")
print(f"Train batches      : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")
print(f"Test batches       : {len(test_loader)}")


# ---------------------------
# 11️⃣ Training loop
# ---------------------------

epochs = cfg["epochs"]

# IMPORTANTE:
# Il best model viene scelto sulla VALIDATION F1,
# NON sul TEST F1.
best_val_f1 = -1.0

best_model_path = os.path.join(
    save_dir,
    f"best_model_{social_dimension}_TE-GIN_2.pt"
)

history = {
    # TRAIN
    "train_ce": [],
    "train_mse": [],
    "train_loss": [],
    "train_acc": [],
    "train_f1": [],

    # VALIDATION
    "val_ce": [],
    "val_mse": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
}


# ============================================================
# FUNZIONE GENERICA PER VAL / TEST
# ============================================================

def evaluate_model(
    model,
    loader,
    dataset_size,
    device,
    class_weights_tensor,
    alpha
):
    model.eval()

    ce_total = 0.0
    mse_total = 0.0
    loss_total = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for graphs, y in loader:

            y = y.to(device)

            out = model(
                graphs,
                device=device
            )

            ce_loss, mse_loss, loss = ordinal_loss(
                out,
                y,
                class_weights_tensor,
                alpha=alpha
            )

            batch_size = y.size(0)

            ce_total += ce_loss.item() * batch_size
            mse_total += mse_loss.item() * batch_size
            loss_total += loss.item() * batch_size

            preds = out.argmax(dim=1)

            all_preds.append(
                preds.cpu().numpy()
            )

            all_labels.append(
                y.cpu().numpy()
            )

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (
        all_preds == all_labels
    ).mean()

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    ce_avg = ce_total / dataset_size
    mse_avg = mse_total / dataset_size
    loss_avg = loss_total / dataset_size

    return (
        ce_avg,
        mse_avg,
        loss_avg,
        acc,
        f1
    )


# ============================================================
# TRAINING
# ============================================================
num_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)

print("Numero parametri: ", num_parameters)
for epoch in range(epochs):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_ce_total = 0.0
    train_mse_total = 0.0
    train_loss_total = 0.0

    all_train_preds = []
    all_train_labels = []

    for graphs, y in train_loader:

        y = y.to(device)

        optimizer.zero_grad()

        out = model(
            graphs,
            device=device
        )

        ce_loss, mse_loss, loss = ordinal_loss(
            out,
            y,
            class_weights_tensor,
            alpha=cfg["alpha"]
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        batch_size = y.size(0)

        train_ce_total += (
            ce_loss.item() * batch_size
        )

        train_mse_total += (
            mse_loss.item() * batch_size
        )

        train_loss_total += (
            loss.item() * batch_size
        )

        all_train_preds.append(
            out.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        all_train_labels.append(
            y.detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Metriche TRAIN
    # --------------------------------------------------------

    all_train_preds = np.concatenate(
        all_train_preds
    )

    all_train_labels = np.concatenate(
        all_train_labels
    )

    train_acc = (
        all_train_preds == all_train_labels
    ).mean()

    train_f1 = f1_score(
        all_train_labels,
        all_train_preds,
        average="macro",
        zero_division=0
    )

    train_ce_avg = (
        train_ce_total /
        len(train_subset)
    )

    train_mse_avg = (
        train_mse_total /
        len(train_subset)
    )

    train_loss_avg = (
        train_loss_total /
        len(train_subset)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_ce_avg,
        val_mse_avg,
        val_loss_avg,
        val_acc,
        val_f1
    ) = evaluate_model(
        model=model,
        loader=val_loader,
        dataset_size=len(val_subset),
        device=device,
        class_weights_tensor=class_weights_tensor,
        alpha=cfg["alpha"]
    )


    # ========================================================
    # TEST
    # ========================================================
    #
    # Il TEST viene calcolato ad ogni epoca perché lo hai
    # richiesto, MA NON viene usato per scegliere il modello.
    #
    # ========================================================



    # ========================================================
    # SALVATAGGIO HISTORY
    # ========================================================

    # TRAIN
    history["train_ce"].append(
        train_ce_avg
    )

    history["train_mse"].append(
        train_mse_avg
    )

    history["train_loss"].append(
        train_loss_avg
    )

    history["train_acc"].append(
        train_acc
    )

    history["train_f1"].append(
        train_f1
    )

    # VALIDATION
    history["val_ce"].append(
        val_ce_avg
    )

    history["val_mse"].append(
        val_mse_avg
    )

    history["val_loss"].append(
        val_loss_avg
    )

    history["val_acc"].append(
        val_acc
    )

    history["val_f1"].append(
        val_f1
    )



    # ========================================================
    # BEST MODEL
    # ========================================================
    #
    # IMPORTANTE:
    # best model scelto sulla VALIDATION F1.
    #
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_f1": best_val_f1,

                # Salviamo anche lo split utilizzato
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,

                # Config
                "hidden_size": cfg["hidden_size"],
                "gnn_type": cfg["gnn_type"],
                "dropout_prob": cfg["dropout_prob"],
                "noise_std": cfg["noise_std"],
                "num_classes": num_classes,

                # Normalizzazione
                "mean": mean,
                "std": std,

                # Class weights
                "class_weights": class_weights_tensor,

                # Relation/config eventualmente utile
                "cfg": cfg
            },
            best_model_path
        )

        improved = "⭐ BEST"

    else:

        improved = ""


    # ========================================================
    # SCHEDULER
    # ========================================================
    #
    # Anche lo scheduler guarda la VALIDATION LOSS,
    # non la TEST LOSS.
    #
    # ========================================================

    scheduler.step(val_loss_avg)


    # ========================================================
    # PRINT EPOCH
    # ========================================================

    print(
        f"\nEpoch {epoch+1:04d}/{epochs} {improved}\n"

        f"  TRAIN | "
        f"CE: {train_ce_avg:.4f} | "
        f"MSE: {train_mse_avg:.4f} | "
        f"Loss: {train_loss_avg:.4f} | "
        f"Acc: {train_acc*100:.2f}% | "
        f"F1: {train_f1:.4f}\n"

        f"  VAL   | "
        f"CE: {val_ce_avg:.4f} | "
        f"MSE: {val_mse_avg:.4f} | "
        f"Loss: {val_loss_avg:.4f} | "
        f"Acc: {val_acc*100:.2f}% | "
        f"F1: {val_f1:.4f}\n"
    )

(
        test_ce_avg,
        test_mse_avg,
        test_loss_avg,
        test_acc,
        test_f1
    ) = evaluate_model(
        model=model,
        loader=val_loader,
        dataset_size=len(val_subset),
        device=device,
        class_weights_tensor=class_weights_tensor,
        alpha=cfg["alpha"]
    )
print(        f"FINAL RESULTS\n   TEST  | "
        f"CE: {test_ce_avg:.4f} | "
        f"MSE: {test_mse_avg:.4f} | "
        f"Loss: {test_loss_avg:.4f} | "
        f"Acc: {test_acc*100:.2f}% | "
        f"F1: {test_f1:.4f}")


# ============================================================
# SALVA HISTORY
# ============================================================

history_path = os.path.join(
    save_dir,
    f"history_TE-GIN_2_{social_dimension}.pt"
)

torch.save(
    history,
    history_path
)

print("\n" + "=" * 70)
print("TRAINING COMPLETATO")
print("=" * 70)

print(
    f"🏆 Best Validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"💾 Best model salvato in:\n"
    f"{best_model_path}"
)

print(
    f"📊 History salvata in:\n"
    f"{history_path}"
)

# ============================================================
# 12️⃣ PLOT METRICHE
# ============================================================

epochs_range = range(1, len(history["train_f1"]) + 1)


# ------------------------------------------------------------
# F1
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_f1"],
    label="Train F1"
)

plt.plot(
    epochs_range,
    history["val_f1"],
    label="Validation F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Train / Validation / Test F1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train / Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    history["val_acc"],
    label="Validation Accuracy"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train / Validation ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# CE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_ce"],
    label="Train CE"
)

plt.plot(
    epochs_range,
    history["val_ce"],
    label="Validation CE"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy")
plt.title("Train / Validation ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# MSE ORDINALE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_mse"],
    label="Train MSE"
)

plt.plot(
    epochs_range,
    history["val_mse"],
    label="Validation MSE"
)


plt.xlabel("Epoch")
plt.ylabel("Ordinal MSE")
plt.title("Train / Validation MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()