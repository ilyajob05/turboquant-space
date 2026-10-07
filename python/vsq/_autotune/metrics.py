"""Input validation, seeded sampling, exact ground truth and recall.

Shapes: base (n, dim), queries (m, dim), both float32; ground truth and
predictions are (m, k) integer ids into base. Distances are squared L2.
"""

from __future__ import annotations

import math

import numpy as np

_EXACT_CHUNK = 64  # queries per exact-distance block: 64 x n float64 stays small


def as_matrix(X, name: str, dim: int | None = None) -> np.ndarray:
    """X as a finite, C-contiguous float32 (n, dim) matrix (one copy at most)."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (n, dim) array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{name} must be a float array, got dtype {arr.dtype}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"{name} has dim {arr.shape[1]}, expected {dim}")
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or inf")
    return out


def split_sample(n: int, n_sample: int, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint sorted row ids: (base_idx (<= n_sample,), query_idx (n_queries,)).

    Queries are drawn first so they never depend on n_sample; the base takes
    up to n_sample of the remaining rows. Same seed -> same split.
    """
    if n_queries < 1 or n_sample < 1:
        raise ValueError(f"n_sample and n_queries must be >= 1, got {n_sample}, {n_queries}")
    if n_queries >= n:
        raise ValueError(f"n_queries={n_queries} must be smaller than n={n}")
    perm = np.random.default_rng(seed).permutation(n)
    query_idx = np.sort(perm[:n_queries])
    base_idx = np.sort(perm[n_queries:n_queries + n_sample])
    return base_idx, query_idx


def squared_l2(base: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact squared L2, shape (n_query, n_base), float64."""
    if base.ndim != 2 or query.ndim != 2:
        raise ValueError(f"squared_l2 expects matrices, got {base.shape} and {query.shape}")
    if base.shape[1] != query.shape[1]:
        raise ValueError(f"dim mismatch: base {base.shape[1]} vs query {query.shape[1]}")
    b = np.asarray(base, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    if not np.isfinite(b).all() or not np.isfinite(q).all():
        raise ValueError("squared_l2 input has a non-finite value")
    return (np.einsum("md,md->m", q, q)[:, None] + np.einsum("nd,nd->n", b, b)[None, :]
            - 2.0 * (q @ b.T))


def topk_indices(dist: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest values per row, nearest first. (m, k) int64."""
    if dist.ndim != 2:
        raise ValueError(f"topk expects a matrix, got {dist.shape}")
    n = dist.shape[1]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    if k == n:
        return np.argsort(dist, axis=1, kind="stable").astype(np.int64)
    part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(dist.shape[0])[:, None]
    order = np.argsort(dist[rows, part], axis=1, kind="stable")
    return part[rows, order].astype(np.int64)


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int,
               chunk: int = _EXACT_CHUNK) -> np.ndarray:
    """Exact squared-L2 top-k ids, (m, k) int64, computed in query chunks."""
    if k > base.shape[0]:
        raise ValueError(f"k={k} exceeds n_base={base.shape[0]}")
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        block = queries[start:start + chunk]
        out[start:start + block.shape[0]] = topk_indices(squared_l2(base, block), k)
    return out


def recall_at(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of the true top-k ids present in the predicted top-k."""
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"recall rows {pred.shape[0]} vs {gt.shape[0]}")
    if k > pred.shape[1] or k > gt.shape[1]:
        raise ValueError(f"recall k={k} exceeds stored {pred.shape[1]} / {gt.shape[1]}")
    hits = (pred[:, :k, None] == gt[:, None, :k]).any(axis=2).sum()
    return float(hits) / (pred.shape[0] * k)


recall_at_k = recall_at


def recall_ci95(recall: float, m: int, k: int) -> tuple[float, float]:
    """95 % normal interval of a recall over m queries x k neighbours."""
    half = 1.96 * math.sqrt(max(recall * (1.0 - recall), 0.0) / max(m * k, 1))
    return max(0.0, recall - half), min(1.0, recall + half)


def recall_tie_tolerance(recall: float, m: int, k: int) -> float:
    """eps_r = max(0.002, two binomial standard errors) for recall ties."""
    return max(0.002, 2.0 * math.sqrt(max(recall * (1.0 - recall), 0.0) / max(m * k, 1)))
