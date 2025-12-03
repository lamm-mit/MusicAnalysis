#!/usr/bin/env python3
"""
scales_vae_cli.py

End-to-end CLI to:
  1) Enumerate all 12-TET scales under simple constraints.
  2) Compute multiple scale metrics:
       - Zeitler defect (missing fifths)
       - Geometric defects: evenness_defect, arrangement_defect,
         unique_intervals_defect, composite_defect
       - Simple Shannon entropy (step-size distribution) + normalized entropy
       - Simple LZ76-style sequence complexity
  3) Train a small Bernoulli VAE on the binary scale vectors.
  4) Export latent codes for all scales.
  5) Generate a range of latent-space plots colored by different metrics.
  6) Optionally push the dataset to Hugging Face Hub via `datasets`.

Default invocation (no special flags) will:
  - Build the dataset in <out_dir>/dataset/scales_dataset.csv
  - Train a 2D VAE
  - Save latents to <out_dir>/vae/latents.csv
  - Produce a set of .svg plots in <out_dir>/plots

Example:
  python scales_vae_cli.py --out-dir out_scales_vae --hf-dataset-id lamm-mit/scales-12tet-defects

Notes:
  - This script is self-contained; it does NOT depend on your previous
    music_analysis.* utilities.
  - Entropy/LZ implementations are intentionally simple but adequate for
    structural exploration of the latent space.
"""

import argparse
import os
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import Dataset as HFDataset  # type: ignore
    _HAS_HF = True
except Exception:
    HFDataset = None  # type: ignore
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Scale representation and basic utilities
# ---------------------------------------------------------------------------

def pcs_from_mask(mask: int, n_pcs: int = 12) -> List[int]:
    """Return sorted list of pitch classes present in a bit-mask."""
    return [i for i in range(n_pcs) if (mask >> i) & 1]


def mask_from_pcs(pcs: List[int], n_pcs: int = 12) -> int:
    """Return integer mask for a list of pitch classes."""
    m = 0
    for p in pcs:
        m |= (1 << (p % n_pcs))
    return m


def max_circular_gap(pcs: List[int], n_pcs: int = 12) -> int:
    """Maximum circular step between consecutive pcs on the pitch-class circle."""
    if len(pcs) <= 1:
        return n_pcs
    pcs = sorted(pcs)
    gaps = [b - a for a, b in zip(pcs, pcs[1:])]
    gaps.append((pcs[0] + n_pcs) - pcs[-1])  # wrap-around
    return max(gaps)


def step_vector_from_pcs(pcs: List[int], n_pcs: int = 12) -> List[int]:
    """Circular step vector g = [step_0, ..., step_{k-1}] whose sum = n_pcs."""
    if len(pcs) <= 1:
        return []
    pcs = sorted(pcs)
    steps = [b - a for a, b in zip(pcs, pcs[1:])]
    steps.append((pcs[0] + n_pcs) - pcs[-1])
    return steps


# ---------------------------------------------------------------------------
# Zeitler defect and simple entropy
# ---------------------------------------------------------------------------

def zeitler_defect(pcs: List[int], n_pcs: int = 12) -> int:
    """
    Zeitler-style imperfection count:
      # of degrees whose perfect fifth (p + 7 mod n_pcs) is not in the scale.
    """
    s = set(pcs)
    imperfections = 0
    for p in pcs:
        if (p + 7) % n_pcs not in s:
            imperfections += 1
    return imperfections


def shannon_entropy_bits_step_distribution(steps: List[int]) -> float:
    """
    Shannon entropy (bits) of the interval-size distribution, as in the Zeitler blog script.
    """
    if not steps:
        return 0.0
    counts: Dict[int, int] = {}
    for st in steps:
        counts[st] = counts.get(st, 0) + 1
    total = float(len(steps))
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def normalized_entropy_bits(steps: List[int]) -> float:
    """
    Normalized Shannon entropy in [0,1].

    We use the same step-distribution entropy as above but divide by the
    theoretical maximum entropy for this length if all step sizes were equally
    likely (i.e., log2(K) where K is the number of distinct step sizes).
    """
    if not steps:
        return 0.0
    H = shannon_entropy_bits_step_distribution(steps)
    distinct = len(set(steps))
    if distinct <= 1:
        # only one step size -> effectively zero entropy
        return 0.0
    H_max = math.log2(distinct)
    return float(H / H_max) if H_max > 0 else 0.0


# ---------------------------------------------------------------------------
# LZ76-style sequence complexity
# ---------------------------------------------------------------------------

def lz76_complexity(sequence: List[int]) -> int:
    """
    Very simple LZ76 complexity for a sequence of discrete symbols.

    Returns the number of distinct phrases parsed by the LZ76 incremental
    parsing algorithm. This is NOT optimized, but fine for short sequences.
    """
    if not sequence:
        return 0
    # Convert to a string of tokens to simplify
    s = ",".join(str(x) for x in sequence)
    i = 0
    c = 1
    while i < len(s):
        l = 1
        # extend substring until it is new
        while i + l <= len(s) and s[i:i + l] in s[:i]:
            l += 1
        i += l
        if i < len(s):
            c += 1
    return c


def lz76_normalized(sequence: List[int]) -> float:
    """
    Normalized LZ76 complexity in [0,1], roughly c / L where L is sequence length.

    This is a heuristic normalization sufficient for comparing scales of
    similar size; exact theoretical normalization is more involved and not
    needed here.
    """
    if not sequence:
        return 0.0
    c = lz76_complexity(sequence)
    L = len(sequence)
    return float(c) / float(L)


# ---------------------------------------------------------------------------
# Geometric defect metrics (your definitions)
# ---------------------------------------------------------------------------

def rotate_list(lst: List[int], tau: int) -> List[int]:
    """Cyclically rotate a list by tau positions."""
    if not lst:
        return lst
    tau %= len(lst)
    return lst[tau:] + lst[:tau]


def cosine_similarity(a: List[int], b: List[int]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def arrangement_defect(steps: List[int]) -> float:
    """
    Arrangement defect in [0,1].

    0  = perfectly palindromic under some rotation of the reversed step vector.
    1  = maximally asymmetric (lowest attainable cosine similarity).
    """
    k = len(steps)
    if k == 0:
        return 0.0
    rev = list(reversed(steps))
    best = -1.0
    for tau in range(k):
        sim = cosine_similarity(steps, rotate_list(rev, tau))
        if sim > best:
            best = sim
    best = max(0.0, min(1.0, best))  # clamp numerical noise
    return 1.0 - best


def compute_per_k_max_std(step_lists_by_k: Dict[int, List[List[int]]], n_pcs: int = 12) -> Dict[int, float]:
    """
    For each k, compute the maximum std(|g - n_pcs/k|) over all observed step vectors g.

    Used to normalize evenness_defect to [0,1] within each k.
    """
    per_k_max_std: Dict[int, float] = {}
    for k, step_list in step_lists_by_k.items():
        if k <= 1:
            per_k_max_std[k] = 1.0
            continue
        mu = float(n_pcs) / float(k)
        max_std = 0.0
        for steps in step_list:
            arr = np.asarray(steps, dtype=float)
            std = float(np.sqrt(np.mean((arr - mu) ** 2)))
            if std > max_std:
                max_std = std
        per_k_max_std[k] = max_std if max_std > 0 else 1.0
    return per_k_max_std


def evenness_defect(steps: List[int], k: int, per_k_max_std_cache: Dict[int, float], n_pcs: int = 12) -> float:
    """
    Evenness defect in [0,1].

    0 = perfectly even spacing (all steps == n_pcs/k),
    1 = the most uneven step pattern observed for this k.
    """
    if k <= 1:
        return 0.0
    mu = float(n_pcs) / float(k)
    arr = np.asarray(steps, dtype=float)
    std = float(np.sqrt(np.mean((arr - mu) ** 2)))
    max_std = per_k_max_std_cache.get(k, None)
    if max_std is None or max_std == 0:
        return 0.0
    return std / max_std


def unique_intervals_defect(steps: List[int], k: int) -> float:
    """
    Unique-intervals defect in [0,1].

    0 = only one distinct step size,
    1 = as many distinct step sizes as possible (capped by min(k, 6)).
    """
    if k <= 1:
        return 0.0
    distinct = len(set(steps))
    max_distinct = min(k, 6)
    if max_distinct <= 1:
        return 0.0
    return float(distinct - 1) / float(max_distinct - 1)


def composite_defect(even: float, arrange: float, uniq: float,
                     w1: float = 0.5, w2: float = 0.3, w3: float = 0.2) -> float:
    """
    Weighted composite defect in [0,1] (assuming inputs in [0,1]).
    """
    return float(w1 * even + w2 * arrange + w3 * uniq)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

@dataclass
class ScaleRecord:
    mask: int
    n_pcs: int
    pcs: List[int]
    steps: List[int]
    k: int
    zeitler_defect: int
    entropy_bits: float
    entropy_norm: float
    lz_norm: float
    evenness_defect: float
    arrangement_defect: float
    unique_intervals_defect: float
    composite_defect: float


def enumerate_scales_with_metrics(
    n_pcs: int = 12,
    min_k: int = 3,
    max_k: Optional[int] = None,
    require_root: bool = True,
    max_gap: Optional[int] = 4,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> List[ScaleRecord]:
    """
    Enumerate all scales satisfying constraints and compute metrics.

    Returns list[ScaleRecord].
    """
    if max_k is None or max_k > n_pcs:
        max_k = n_pcs

    valid_scales: List[Tuple[int, List[int], List[int]]] = []

    # Enumerate all non-empty masks
    for mask in range(1, 1 << n_pcs):
        if require_root and not (mask & 1):
            continue
        pcs = pcs_from_mask(mask, n_pcs=n_pcs)
        k = len(pcs)
        if k < min_k or k > max_k:
            continue
        if max_gap is not None and max_gap > 0:
            if max_circular_gap(pcs, n_pcs=n_pcs) > max_gap:
                continue
        steps = step_vector_from_pcs(pcs, n_pcs=n_pcs)
        if not steps:
            continue
        valid_scales.append((mask, pcs, steps))

    if not valid_scales:
        raise RuntimeError("No valid scales found under the given constraints.")

    # Group step vectors by k for evenness normalization
    by_k_steps: Dict[int, List[List[int]]] = {}
    for _, pcs, steps in valid_scales:
        kk = len(pcs)
        by_k_steps.setdefault(kk, []).append(steps)

    per_k_max_std = compute_per_k_max_std(by_k_steps, n_pcs=n_pcs)

    # Build records
    records: List[ScaleRecord] = []
    for mask, pcs, steps in valid_scales:
        k = len(pcs)
        z_def = zeitler_defect(pcs, n_pcs=n_pcs)
        H_bits = shannon_entropy_bits_step_distribution(steps)
        H_norm = normalized_entropy_bits(steps)
        lz_norm = lz76_normalized(steps)
        even_def = evenness_defect(steps, k, per_k_max_std, n_pcs=n_pcs)
        arr_def = arrangement_defect(steps)
        uniq_def = unique_intervals_defect(steps, k)
        comp_def = composite_defect(even_def, arr_def, uniq_def, w1=w1, w2=w2, w3=w3)

        rec = ScaleRecord(
            mask=mask,
            n_pcs=n_pcs,
            pcs=list(pcs),
            steps=list(steps),
            k=k,
            zeitler_defect=z_def,
            entropy_bits=float(H_bits),
            entropy_norm=float(H_norm),
            lz_norm=float(lz_norm),
            evenness_defect=float(even_def),
            arrangement_defect=float(arr_def),
            unique_intervals_defect=float(uniq_def),
            composite_defect=float(comp_def),
        )
        records.append(rec)

    return records


def records_to_dataframe(records: List[ScaleRecord]) -> pd.DataFrame:
    """
    Convert list[ScaleRecord] to a pandas DataFrame, including a 0/1 vector
    representation of each scale (x_0,...,x_{n_pcs-1}).
    """
    rows = []
    for r in records:
        d = asdict(r)
        # Add binary vector x
        x = [(d["mask"] >> i) & 1 for i in range(d["n_pcs"])]
        for i, val in enumerate(x):
            d[f"x_{i}"] = int(val)
        # Convert pcs/steps to JSON strings for CSV-friendliness
        d["pcs_json"] = json.dumps(d["pcs"])
        d["steps_json"] = json.dumps(d["steps"])
        rows.append(d)
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# VAE model
# ---------------------------------------------------------------------------

class ScalesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_pcs: int = 12):
        self.n_pcs = n_pcs
        xs = []
        for _, row in df.iterrows():
            xs.append([row[f"x_{i}"] for i in range(n_pcs)])
        self.x = torch.tensor(xs, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class VAE(nn.Module):
    def __init__(self, input_dim: int = 12, latent_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Bernoulli parameters
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Bernoulli reconstruction loss + beta * KL.
    """
    # Bernoulli NLL; add a small epsilon to avoid log(0)
    eps = 1e-7
    recon_x = torch.clamp(recon_x, eps, 1.0 - eps)
    bce = - (x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x)).sum(dim=1).mean()
    # KL divergence between q(z|x) and N(0,I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return bce + beta * kl


def train_vae(
    df: pd.DataFrame,
    n_pcs: int,
    latent_dim: int = 2,
    hidden_dim: int = 64,
    batch_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: Optional[str] = None,
) -> Tuple[VAE, pd.DataFrame]:
    """
    Train a VAE on the binary scale vectors.

    Returns (model, latents_df) where latents_df contains z1,z2,... for each row of df.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ScalesDataset(df, n_pcs=n_pcs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VAE(input_dim=n_pcs, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            avg_loss = total_loss / max(1, n_batches)
            print(f"[VAE] Epoch {epoch:4d}/{epochs}: loss = {avg_loss:.4f}")

    # Compute latents (mu) for all data
    model.eval()
    with torch.no_grad():
        all_x = dataset.x.to(device)
        mu, logvar = model.encode(all_x)
        z = mu.cpu().numpy()

    lat_cols = {}
    for i in range(latent_dim):
        lat_cols[f"z_{i}"] = z[:, i]

    latents_df = df.copy().reset_index(drop=True)
    for k, v in lat_cols.items():
        latents_df[k] = v

    return model, latents_df


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_latent_scatter_color_by(
    latents_df: pd.DataFrame,
    out_path: str,
    color_col: str,
    title: str,
    x_col: str = "z_0",
    y_col: str = "z_1",
    alpha: float = 0.5,
) -> None:
    """
    Scatter of latent points colored by a scalar column.
    """
    plt.figure(figsize=(6, 5))
    x = latents_df[x_col].values
    y = latents_df[y_col].values
    c = latents_df[color_col].values
    sc = plt.scatter(x, y, c=c, alpha=alpha)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.colorbar(sc, label=color_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latent_scatter_discrete_k(
    latents_df: pd.DataFrame,
    out_path: str,
    x_col: str = "z_0",
    y_col: str = "z_1",
    alpha: float = 0.5,
) -> None:
    """
    Scatter where color encodes k (scale size).
    """
    plt.figure(figsize=(6, 5))
    x = latents_df[x_col].values
    y = latents_df[y_col].values
    c = latents_df["k"].values
    sc = plt.scatter(x, y, c=c, alpha=alpha)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Latent space colored by k (number of notes)")
    plt.colorbar(sc, label="k")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_z_vs_metric(
    latents_df: pd.DataFrame,
    out_path: str,
    z_col: str,
    metric_col: str,
    title: Optional[str] = None,
) -> None:
    """
    2D scatter of one latent coordinate vs a metric; also print correlation.
    """
    x = latents_df[z_col].values
    y = latents_df[metric_col].values
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.4, s=8)
    plt.xlabel(z_col)
    plt.ylabel(metric_col)
    if title is None:
        title = f"{z_col} vs {metric_col}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Simple correlation print to console
    if len(x) > 1:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denom > 0:
            r = float(np.dot(x_centered, y_centered) / denom)
            print(f"Correlation({z_col}, {metric_col}) = {r:.3f}")


def make_default_plots(latents_df: pd.DataFrame, plots_dir: str) -> None:
    """
    Produce a range of default plots in the latent space.
    """
    ensure_dir(plots_dir)

    # 1) Latent scatter colored by k
    plot_latent_scatter_discrete_k(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_k.svg"),
    )

    # 2) Colored by Zeitler defect
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_zeitler_defect.svg"),
        color_col="zeitler_defect",
        title="Latent space colored by Zeitler defect",
    )

    # 3) Colored by composite defect
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_composite_defect.svg"),
        color_col="composite_defect",
        title="Latent space colored by composite defect",
    )

    # 4) Colored by evenness_defect
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_evenness_defect.svg"),
        color_col="evenness_defect",
        title="Latent space colored by evenness_defect",
    )

    # 5) Colored by arrangement_defect
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_arrangement_defect.svg"),
        color_col="arrangement_defect",
        title="Latent space colored by arrangement_defect",
    )

    # 6) Colored by unique_intervals_defect
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_unique_intervals_defect.svg"),
        color_col="unique_intervals_defect",
        title="Latent space colored by unique_intervals_defect",
    )

    # 7) Colored by normalized entropy
    plot_latent_scatter_color_by(
        latents_df,
        os.path.join(plots_dir, "latent_scatter_by_entropy_norm.svg"),
        color_col="entropy_norm",
        title="Latent space colored by normalized entropy",
    )

    # 8) z_0 vs key metrics
    for metric in [
        "k",
        "zeitler_defect",
        "composite_defect",
        "entropy_norm",
        "lz_norm",
        "evenness_defect",
        "arrangement_defect",
        "unique_intervals_defect",
    ]:
        plot_z_vs_metric(
            latents_df,
            os.path.join(plots_dir, f"z0_vs_{metric}.svg"),
            z_col="z_0",
            metric_col=metric,
        )

    # 9) z_1 vs key metrics (if 2D latent)
    if "z_1" in latents_df.columns:
        for metric in [
            "k",
            "zeitler_defect",
            "composite_defect",
            "entropy_norm",
            "lz_norm",
            "evenness_defect",
            "arrangement_defect",
            "unique_intervals_defect",
        ]:
            plot_z_vs_metric(
                latents_df,
                os.path.join(plots_dir, f"z1_vs_{metric}.svg"),
                z_col="z_1",
                metric_col=metric,
            )


# ---------------------------------------------------------------------------
# Hugging Face push
# ---------------------------------------------------------------------------

def push_dataset_to_hf(
    df: pd.DataFrame,
    repo_id: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Push the dataset (scales + metrics) to Hugging Face Hub using `datasets`.

    repo_id: e.g. "lamm-mit/scales-12tet-defects"
    hf_token: passed explicitly or read from HF_TOKEN / HUGGINGFACE_TOKEN env.
    """
    if not _HAS_HF:
        raise RuntimeError("datasets library is not installed; cannot push to Hugging Face.")

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token is None:
        raise RuntimeError("HF token not provided (via --hf-token or env HF_TOKEN/HUGGINGFACE_TOKEN).")

    # Convert to a HF Dataset
    # For list-like columns, convert JSON strings back to Python lists so HF
    # treats them as sequences.
    df_for_hf = df.copy()
    if "pcs_json" in df_for_hf.columns:
        df_for_hf["pcs"] = df_for_hf["pcs_json"].apply(json.loads)
    if "steps_json" in df_for_hf.columns:
        df_for_hf["steps"] = df_for_hf["steps_json"].apply(json.loads)

    # Drop the JSON helper columns; keep mask/x_* etc.
    drop_cols = [c for c in ["pcs_json", "steps_json"] if c in df_for_hf.columns]
    if drop_cols:
        df_for_hf = df_for_hf.drop(columns=drop_cols)

    hf_dataset = HFDataset.from_pandas(df_for_hf, preserve_index=False)  # type: ignore
    print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    hf_dataset.push_to_hub(repo_id=repo_id, token=hf_token)  # type: ignore
    print("Push complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Enumerate 12-TET scales, train a VAE, plot latents, and optionally push dataset to Hugging Face.")
    ap.add_argument("--out-dir", type=str, default="out_scales_vae", help="Base output directory.")
    ap.add_argument("--n-pcs", type=int, default=12, help="Number of pitch classes (default 12).")
    ap.add_argument("--min-k", type=int, default=3, help="Minimum scale size.")
    ap.add_argument("--max-k", type=int, default=12, help="Maximum scale size.")
    ap.add_argument("--no-require-root", action="store_true", help="Do NOT require pc=0 to be in the scale.")
    ap.add_argument("--max-gap", type=int, default=4, help="Max circular step; set to 0 or use --no-gap-filter to disable.")
    ap.add_argument("--no-gap-filter", action="store_true", help="Disable the max-gap filter entirely.")
    ap.add_argument("--w1", type=float, default=0.5, help="Weight for evenness_defect in composite defect.")
    ap.add_argument("--w2", type=float, default=0.3, help="Weight for arrangement_defect in composite defect.")
    ap.add_argument("--w3", type=float, default=0.2, help="Weight for unique_intervals_defect in composite defect.")

    # VAE hyperparameters
    ap.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality of the VAE.")
    ap.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width in the VAE.")
    ap.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for VAE training.")
    ap.add_argument("--epochs", type=int, default=200, help="Number of VAE training epochs.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for VAE training.")
    ap.add_argument("--beta", type=float, default=1.0, help="Beta for beta-VAE (KL weight).")
    ap.add_argument("--device", type=str, default=None, help="Torch device (cpu or cuda). Default: auto.")

    # Control which stages run
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset building (reuse existing CSV).")
    ap.add_argument("--skip-train", action="store_true", help="Skip VAE training + latent export (reuse existing latents CSV).")
    ap.add_argument("--skip-plots", action="store_true", help="Skip plotting.")

    # Hugging Face
    ap.add_argument("--hf-dataset-id", type=str, default=None, help="Hugging Face dataset repo_id to push to (e.g. 'user/scales-12tet').")
    ap.add_argument("--hf-token", type=str, default=None, help="Hugging Face token; if omitted, read HF_TOKEN/HUGGINGFACE_TOKEN.")

    args = ap.parse_args()

    out_dir = args.out_dir
    dataset_dir = os.path.join(out_dir, "dataset")
    vae_dir = os.path.join(out_dir, "vae")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(dataset_dir)
    ensure_dir(vae_dir)
    ensure_dir(plots_dir)

    dataset_csv_path = os.path.join(dataset_dir, "scales_dataset.csv")
    latents_csv_path = os.path.join(vae_dir, "latents.csv")
    vae_model_path = os.path.join(vae_dir, "vae_model.pt")

    # ------------------------------------------------------------------
    # 1) Build dataset (if not skipped)
    # ------------------------------------------------------------------
    if not args.skip_build or not os.path.exists(dataset_csv_path):
        require_root = not args.no_require_root
        max_gap = None if args.no_gap_filter or args.max_gap <= 0 else args.max_gap

        print("Enumerating scales and computing metrics...")
        records = enumerate_scales_with_metrics(
            n_pcs=args.n_pcs,
            min_k=args.min_k,
            max_k=args.max_k,
            require_root=require_root,
            max_gap=max_gap,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
        )
        df = records_to_dataframe(records)
        df.to_csv(dataset_csv_path, index=False)
        print(f"Saved dataset with {len(df)} scales to {dataset_csv_path}")
    else:
        print(f"Skipping build; loading existing dataset from {dataset_csv_path}")
        df = pd.read_csv(dataset_csv_path)

    # ------------------------------------------------------------------
    # 2) Train VAE and export latents (if not skipped)
    # ------------------------------------------------------------------
    if not args.skip_train or not os.path.exists(latents_csv_path):
        print("Training VAE on scale vectors...")
        model, latents_df = train_vae(
            df,
            n_pcs=args.n_pcs,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            beta=args.beta,
            device=args.device,
        )
        # Save model
        torch.save(model.state_dict(), vae_model_path)
        print(f"Saved VAE model to {vae_model_path}")
        # Save latents
        latents_df.to_csv(latents_csv_path, index=False)
        print(f"Saved latents to {latents_csv_path}")
    else:
        print(f"Skipping VAE training; loading existing latents from {latents_csv_path}")
        latents_df = pd.read_csv(latents_csv_path)

    # ------------------------------------------------------------------
    # 3) Plots
    # ------------------------------------------------------------------
    if not args.skip_plots:
        print("Generating default latent-space plots...")
        make_default_plots(latents_df, plots_dir)
        print(f"Plots written to {plots_dir}")

    # ------------------------------------------------------------------
    # 4) Optional push to Hugging Face
    # ------------------------------------------------------------------
    if args.hf_dataset_id is not None:
        print("Pushing dataset to Hugging Face Hub...")
        push_dataset_to_hf(df, args.hf_dataset_id, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
