"""One search interface over the four index kinds the autotuner can choose.

    turboquant flat      TurboQuantSpace codes, distance_1_to_n + top-k
    turboquant fastscan  TurboQuantFastScan over the same codes (4-bit)
    rabitq flat          RaBitQSpace codes, distance_1_to_n + top-k
    rabitq fastscan      RaBitQFastScan, built from the raw vectors

search(q, k) -> (ids (k,) uint32, dists (k,) float32), nearest first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .._vsq import RaBitQFastScan, RaBitQSpace, TurboQuantFastScan, TurboQuantSpace
from .metrics import as_matrix
from .types import QuantizerConfig


@dataclass(frozen=True)
class BuildStats:
    """Wall seconds of train() and of encoding / index construction for n rows."""

    n: int
    train_s: float
    encode_s: float

    @property
    def encode_vps(self) -> float:
        return self.n / max(self.encode_s, 1e-9)


def make_space(config: QuantizerConfig, dim: int):
    """Untrained space for ``config`` (TurboQuantSpace or RaBitQSpace)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if config.family == "turboquant":
        return TurboQuantSpace(
            dim, config.bits, centering=config.centering, n_clusters=config.n_clusters,
            rotation_rounds=config.rotation_rounds, rot_seed=config.rot_seed,
            num_threads=config.num_threads)
    return RaBitQSpace(
        dim, rot_seed=config.rot_seed, bits=config.bits, encode_mode=config.encode_mode,
        query_bits=config.query_bits, num_threads=config.num_threads)


def train_rows(X: np.ndarray, rows: int, seed: int) -> np.ndarray:
    """Seeded, sorted subsample of at most ``rows`` rows (all rows if fewer)."""
    if X.shape[0] <= rows:
        return X
    idx = np.sort(np.random.default_rng(seed).choice(X.shape[0], rows, replace=False))
    return np.ascontiguousarray(X[idx])


def train_space(space, config: QuantizerConfig, X: np.ndarray) -> None:
    """Fit centering (TurboQuant ivf/mean) or the centroid (RaBitQ, always)."""
    sample = train_rows(X, config.train_rows, config.train_seed)
    if config.family == "turboquant":
        if config.centering != "none":
            space.train(sample, seed=config.train_seed, iters=config.train_iters)
    else:
        space.train(sample)


class QuantizedIndex:
    """Searchable index over n vectors of one configuration."""

    def __init__(self, config: QuantizerConfig, space, codes: np.ndarray | None, fastscan) -> None:
        self.config = config
        self.space = space
        self.codes = codes  # (n, code_size) uint8; None for RaBitQ FastScan
        self._fs = fastscan
        self._n = len(fastscan) if fastscan is not None else codes.shape[0]
        self._dim = int(space.dim())

    def __len__(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def code_size_bytes(self) -> int:
        """Stored bytes per vector of the underlying space."""
        return int(self.space.code_size_bytes())

    def _query(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (self._dim,):
            raise ValueError(f"query must have shape ({self._dim},), got {q.shape}")
        return np.ascontiguousarray(q)

    def search_stats(self, q, k: int) -> tuple[np.ndarray, np.ndarray, int]:
        """(ids, dists, refined): refined = codes re-scored with the full code."""
        if not 1 <= k <= self._n:
            raise ValueError(f"k must be in 1..{self._n}, got {k}")
        q = self._query(q)
        cfg = self.config
        if cfg.index == "flat":
            d = self.space.distance_1_to_n(q, self.codes)
            if k == self._n:
                ids = np.argsort(d, kind="stable")
            else:
                part = np.argpartition(d, k - 1)[:k]
                ids = part[np.argsort(d[part], kind="stable")]
            return ids.astype(np.uint32), d[ids].astype(np.float32), self._n
        if cfg.family == "turboquant":
            rerank = cfg.rerank or 4
            ids, dists = self._fs.search(q, k, rerank=rerank)
            return ids, dists, min(k * rerank, self._n)
        ids, dists, refined = self._fs.search(q, k)
        return ids, dists, int(refined)

    def search(self, q, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(ids (k,) uint32, dists (k,) float32), nearest first."""
        ids, dists, _ = self.search_stats(q, k)
        return ids, dists

    def search_batch(self, Q, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Loop of single-query searches: (ids (m, k), dists (m, k))."""
        Q = as_matrix(Q, "Q", self._dim)
        ids = np.empty((Q.shape[0], k), dtype=np.uint32)
        dists = np.empty((Q.shape[0], k), dtype=np.float32)
        for i, q in enumerate(Q):
            ids[i], dists[i] = self.search(q, k)
        return ids, dists

    def tiled(self, n_rows: int) -> QuantizedIndex:
        """Same codes repeated to n_rows (timing only: the scan cost of flat and
        TurboQuant FastScan does not depend on the code values)."""
        if self.codes is None:
            raise ValueError("RaBitQ FastScan is built from vectors and cannot be tiled")
        if n_rows < 1:
            raise ValueError(f"n_rows must be >= 1, got {n_rows}")
        reps = -(-n_rows // self.codes.shape[0])
        codes = np.ascontiguousarray(np.tile(self.codes, (reps, 1))[:n_rows])
        fs = TurboQuantFastScan(self.space, codes) if self._fs is not None else None
        return QuantizedIndex(self.config, self.space, codes, fs)


def build_index(config: QuantizerConfig, X, *, trained_space=None) -> tuple[QuantizedIndex, BuildStats]:
    """Train (unless trained_space is given), encode X (n, dim) float32, wrap."""
    X = as_matrix(X, "X")
    config.validate()
    train_s = 0.0
    space = trained_space
    if space is None:
        space = make_space(config, X.shape[1])
        start = time.perf_counter()
        train_space(space, config, X)
        train_s = time.perf_counter() - start
    start = time.perf_counter()
    if config.family == "rabitq" and config.index == "fastscan":
        eps0 = config.eps0 if config.eps0 is not None else 1.9
        index = QuantizedIndex(config, space, None, RaBitQFastScan(space, X, eps0=eps0))
    else:
        codes = np.asarray(space.encode_batch(X), dtype=np.uint8)
        expected = (X.shape[0], int(space.code_size_bytes()))
        if codes.shape != expected:
            raise RuntimeError(f"encode_batch returned {codes.shape}, expected {expected}")
        fs = TurboQuantFastScan(space, codes) if config.index == "fastscan" else None
        index = QuantizedIndex(config, space, codes, fs)
    return index, BuildStats(n=X.shape[0], train_s=train_s,
                             encode_s=time.perf_counter() - start)
