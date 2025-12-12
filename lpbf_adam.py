"""
LPBF Ti-6Al-4V parameter optimization using a neural-network surrogate with Adam.

Main training data file:
    lpbf_ti64_data.xlsx  (Sheet1)
Columns (must match):
    hatch_spacing, laser_power, scan_speed, layer_thickness,
    yield_strength, elastic_modulus

Suggestion save to:

    "To be tested.xlsx"

Columns in that file:
    hatch_spacing, laser_power, scan_speed, layer_thickness,
    yield_strength, elastic_modulus
(yield_strength & elastic_modulus are PREDICTED, rounded to 2 decimals)

Per run, we produce:
    - 2 recommendations biased toward high yield_strength
    - 1 recommendation biased toward high elastic_modulus
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

print("=== LPBF Adam Surrogate v1.3 (2×YS, 1×E, randomized) ===")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DATA_PATH = "lpbf_ti64_data.xlsx"   # training data
SHEET_NAME = "Sheet1"

TO_TEST_PATH = "To be tested.xlsx"  # file storing recommended parameters

FEATURE_COLS = [
    "hatch_spacing",
    "laser_power",
    "scan_speed",
    "layer_thickness",
]

TARGET_COLS = [
    "yield_strength",
    "elastic_modulus",
]

BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 300
VAL_SPLIT = 0.2
SEED = 42  # used for training reproducibility; recs will use their own RNG

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class LPBFDataset(Dataset):
    def __init__(self, data_path, feature_cols, target_cols, sheet_name="Sheet1"):
        df = pd.read_excel(data_path, sheet_name=sheet_name, header=0)

        print("Main Excel columns:", list(df.columns))
        print("First rows in main sheet:\n", df.head())

        missing = [c for c in feature_cols + target_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in Excel: {missing}\n"
                f"Current columns: {list(df.columns)}"
            )

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)

        self.X_raw = X
        self.y_raw = y

        self.X_mean = X.mean(axis=0, keepdims=True)
        self.X_std = X.std(axis=0, keepdims=True) + 1e-8

        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-8

        self.X_norm = (X - self.X_mean) / self.X_std
        self.y_norm = (y - self.y_mean) / self.y_std

    def __len__(self):
        return self.X_norm.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X_norm[idx])
        y = torch.from_numpy(self.y_norm[idx])
        return x, y


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SurrogateNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(model, train_loader, val_loader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    has_val = val_loader is not None and len(val_loader.dataset) > 0

    best_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        if has_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = train_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_state_dict = model.state_dict()

        if epoch % 20 == 0 or epoch == 1:
            if has_val:
                print(
                    f"Epoch {epoch:4d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}"
                )
            else:
                print(
                    f"Epoch {epoch:4d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4e}"
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("Training complete. Best loss:", best_loss)
    return model


# ------------------------------------------------------------
# Candidate sampling + objective-specific selection
# ------------------------------------------------------------

def sample_candidates_and_score(model, dataset, objective="yield", num_candidates=400):
    """
    Sample candidate parameter sets within a padded range around existing data,
    evaluate with the surrogate, and return:
        candidates_raw (num_candidates, 4)
        y_pred_np      (num_candidates, 2)  [YS, E]
        scores         (num_candidates,)    [depending on objective]
    objective: "yield" or "E"
    """
    assert objective in ("yield", "E")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_raw = dataset.X_raw  # shape (N, 4)
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    x_range = x_max - x_min

    # Avoid zero range (if all data same)
    x_range = np.where(x_range < 1e-6, 1.0, x_range)

    # +/- 10% padding
    lower = x_min - 0.1 * x_range
    upper = x_max + 0.1 * x_range

    num_features = len(FEATURE_COLS)

    # Use a fresh RNG so each run is different, independent of global seed
    rng = np.random.default_rng()
    candidates_raw = rng.uniform(
        low=lower, high=upper, size=(num_candidates, num_features)
    ).astype(np.float32)

    # Normalize for model
    candidates_norm = (candidates_raw - dataset.X_mean) / dataset.X_std

    # Evaluate
    with torch.no_grad():
        x_t = torch.from_numpy(candidates_norm).to(device)
        y_pred_norm = model(x_t)
        y_pred = (
            y_pred_norm * torch.from_numpy(dataset.y_std).to(device)
            + torch.from_numpy(dataset.y_mean).to(device)
        )
        y_pred_np = y_pred.cpu().numpy()

    idx_yield = TARGET_COLS.index("yield_strength")
    idx_E = TARGET_COLS.index("elastic_modulus")

    yields = y_pred_np[:, idx_yield]
    moduli = y_pred_np[:, idx_E]

    if objective == "yield":
        scores = yields  # maximize yield
    else:  # objective == "E"
        scores = moduli  # maximize E

    return candidates_raw, y_pred_np, scores, x_min, x_range


def pick_diverse_top(candidates_raw, y_pred_np, scores, x_min, x_range, n_recs):
    """
    From candidate set with given scores, pick n_recs best ones,
    enforcing diversity in normalized parameter space.
    """
    sorted_idx = np.argsort(-scores)  # descending
    chosen = []
    chosen_indices = []

    # Normalize parameters to [0,1] for distance measure
    norm_for_dist = (candidates_raw - x_min) / (x_range + 1e-8)
    min_norm_dist = 0.4  # ~40% distance to be considered "different"

    for idx in sorted_idx:
        if len(chosen) == 0:
            chosen.append(idx)
            chosen_indices.append(idx)
        else:
            dists = []
            for j in chosen:
                diff = norm_for_dist[idx] - norm_for_dist[j]
                d = np.linalg.norm(diff)
                dists.append(d)
            if np.min(dists) >= min_norm_dist:
                chosen.append(idx)
                chosen_indices.append(idx)
        if len(chosen) >= n_recs:
            break

    # If not enough diverse ones, just take top few
    if len(chosen_indices) < n_recs:
        for idx in sorted_idx:
            if idx not in chosen_indices:
                chosen_indices.append(idx)
            if len(chosen_indices) >= n_recs:
                break

    recs = []
    for idx in chosen_indices[:n_recs]:
        x_vec = candidates_raw[idx]
        y_vec = y_pred_np[idx]
        best_x = {name: float(val) for name, val in zip(FEATURE_COLS, x_vec)}
        best_y = {name: float(val) for name, val in zip(TARGET_COLS, y_vec)}
        recs.append((best_x, best_y))

    return recs


def get_recommendations(model, dataset, n_ys=2, n_E=1, num_candidates=400):
    """
    Generate:
        - n_ys recommendations for high yield strength
        - n_E recommendations for high elastic modulus
    Using separate random candidate sets for each objective.
    """
    # High YS recommendations
    cand_raw_y, y_pred_y, scores_y, x_min_y, x_range_y = sample_candidates_and_score(
        model, dataset, objective="yield", num_candidates=num_candidates
    )
    recs_y = pick_diverse_top(cand_raw_y, y_pred_y, scores_y, x_min_y, x_range_y, n_ys)

    # High E recommendations
    cand_raw_E, y_pred_E, scores_E, x_min_E, x_range_E = sample_candidates_and_score(
        model, dataset, objective="E", num_candidates=num_candidates
    )
    recs_E = pick_diverse_top(cand_raw_E, y_pred_E, scores_E, x_min_E, x_range_E, n_E)

    # Combine (order: 2×YS, 1×E)
    return recs_y + recs_E


# ------------------------------------------------------------
# Write to "To be tested.xlsx"
# ------------------------------------------------------------

def append_recommendations_to_test_sheet(recs, path: str):
    """
    recs: list of (best_x_dict, best_y_dict)
    Writes all as new rows to To be tested.xlsx
    """
    rows = []
    for best_x, best_y in recs:
        row = {
            "hatch_spacing": round(best_x["hatch_spacing"], 4),
            "laser_power": round(best_x["laser_power"], 4),
            "scan_speed": round(best_x["scan_speed"], 4),
            "layer_thickness": round(best_x["layer_thickness"], 4),
            "yield_strength": round(best_y["yield_strength"], 2),     # 2 digits
            "elastic_modulus": round(best_y["elastic_modulus"], 2),   # 2 digits
        }
        rows.append(row)

    new_df = pd.DataFrame(rows)

    abs_path = os.path.abspath(path)
    print("\n========== WRITING RECOMMENDATIONS TO TEST SHEET ==========")
    print(f"Target file: {abs_path}")

    if os.path.exists(path):
        existing = pd.read_excel(path, sheet_name=0)
        combined = pd.concat([existing, new_df], ignore_index=True)
        cols = ["hatch_spacing", "laser_power", "scan_speed",
                "layer_thickness", "yield_strength", "elastic_modulus"]
        combined = combined[cols]
        combined.to_excel(path, index=False)
        print(f"Appended {len(rows)} rows to existing 'To be tested.xlsx'.")
    else:
        new_df.to_excel(path, index=False)
        print(f"Created new 'To be tested.xlsx' with {len(rows)} rows.")

    print("Rows written:")
    print(new_df)
    print("===========================================================\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    dataset = LPBFDataset(
        DATA_PATH,
        FEATURE_COLS,
        TARGET_COLS,
        sheet_name=SHEET_NAME,
    )

    n = len(dataset)
    print(f"\nNumber of data points in main Excel: {n}")

    if n < 5:
        print("Small dataset -> using all data for training, no validation set.")
        train_ds = dataset
        val_ds = None
    else:
        val_size = int(n * VAL_SPLIT)
        train_size = n - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if val_ds is not None else None

    model = SurrogateNet(
        input_dim=len(FEATURE_COLS),
        output_dim=len(TARGET_COLS),
    )
    model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LR)

    print("\n=== Searching for 3 parameter sets (2×YS-focused, 1×E-focused) ===")
    recs = get_recommendations(model, dataset, n_ys=2, n_E=1, num_candidates=400)

    for i, (best_x, best_y) in enumerate(recs, start=1):
        print(f"\n----- Recommendation {i} -----")
        print("Suggested LPBF parameters:")
        for k, v in best_x.items():
            print(f"  {k:16s} = {v:.4f}")
        print("Predicted properties (2 decimals):")
        print(f"  yield_strength   = {best_y['yield_strength']:.2f} MPa")
        print(f"  elastic_modulus  = {best_y['elastic_modulus']:.2f} GPa")

    # Write all 3 to To be tested.xlsx
    append_recommendations_to_test_sheet(recs, TO_TEST_PATH)


if __name__ == "__main__":
    main()
