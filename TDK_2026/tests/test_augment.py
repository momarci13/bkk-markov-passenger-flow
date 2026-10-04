"""Tests for scripts/paper_augment.py (Section XI of the paper)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import paper_augment as pa  # noqa: E402


def _random_layer(n=25, p=0.12, seed=0):
    rng = np.random.default_rng(seed)
    W = np.where(rng.random((n, n)) < p, rng.uniform(10, 100, (n, n)), 0.0)
    np.fill_diagonal(W, 0.0)
    lat = 47.5 + rng.uniform(-0.02, 0.02, n)
    lon = 19.05 + rng.uniform(-0.03, 0.03, n)
    return pa.Layer("test", [f"s{i}" for i in range(n)], lat, lon, csr_matrix(W),
                    rng.uniform(1, 5, n) / rng.uniform(1, 5, n).sum(), 0.05)


def _remove(W, k):
    W = W.tolil(copy=True)
    W[k, :] = 0
    W[:, k] = 0
    D = dijkstra(W.tocsr(), directed=True)
    D[k, :] = np.inf
    D[:, k] = np.inf
    return D


@pytest.mark.parametrize("seed", range(5))
def test_shortcut_update_is_exact(seed):
    """Proposition XI.1: the update equals a full recomputation, also after a
    failure, and is unchanged when the failed node is a link endpoint."""
    L = _random_layer(seed=seed)
    ev = pa.Evaluator(L, np.arange(L.n))
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(3):
        a, b = rng.choice(L.n, 2, replace=False)
        edges.append((int(a), int(b), float(rng.uniform(5, 50))))
    D, Dk = ev.with_edges(edges)
    W = L.W.tolil(copy=True)
    for a, b, c in edges:
        W[a, b] = min(W[a, b], c) if W[a, b] else c
        W[b, a] = min(W[b, a], c) if W[b, a] else c
    W = W.tocsr()
    ref = dijkstra(W, directed=True)
    assert np.array_equal(np.isinf(D), np.isinf(ref))
    fin = np.isfinite(ref)
    assert np.allclose(D[fin], ref[fin], atol=1e-9)
    for k in range(L.n):
        # the failure removes every link incident to k, new ones included
        refk = _remove(W, k)
        fin = np.isfinite(refk)
        assert np.array_equal(np.isinf(Dk[k]), np.isinf(refk))
        assert np.allclose(Dk[k][fin], refk[fin], atol=1e-9)


def test_pendant_extension_leaves_vulnerability_unchanged():
    """Proposition XI.2 on a path graph extended by a pendant vertex."""
    n = 6
    W = np.zeros((n + 1, n + 1))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 60.0
    W[0, n - 1] = W[n - 1, 0] = 200.0          # ring, so there is redundancy
    lat = 47.5 + 0.001 * np.arange(n + 1)
    lon = np.full(n + 1, 19.05)
    p = np.r_[np.ones(n), 0.0]                  # the new station has no demand
    base = pa.Layer("t", [str(i) for i in range(n + 1)], lat, lon,
                    csr_matrix(W), p / p.sum(), 0.05)
    W2 = W.copy()
    W2[2, n] = W2[n, 2] = 60.0                  # attach vertex n at station 2
    ext = pa.Layer("t", base.names, lat, lon, csr_matrix(W2), base.p, 0.05)
    F = np.arange(n)
    V0 = pa.Evaluator(base, F).vuln()
    V1 = pa.Evaluator(ext, F).vuln()
    assert np.allclose(V0, V1)


def test_hub_reinforcement_trap():
    """Proposition XI.3: links that all end at k cannot reduce V_k."""
    L = _random_layer(n=20, p=0.2, seed=3)
    ev = pa.Evaluator(L, np.arange(L.n))
    V0 = ev.vuln()
    k = int(np.argmax(V0))
    others = [j for j in range(L.n) if j != k][:4]
    D, Dk = ev.with_edges([(k, j, 20.0) for j in others])
    V1 = ev.vuln(D, Dk)
    assert V1[k] >= V0[k] - 1e-12


def test_greedy_is_monotone_in_J():
    L = _random_layer(n=18, p=0.18, seed=7)
    ev = pa.Evaluator(L, np.arange(L.n))
    cands = pa.candidates(L, cap_m=5000, no_river=False, min_detour=1.0)
    base, chosen, steps, _ = pa.greedy(ev, cands, 3)
    Js = [base["J"]] + [s["J"] for s in steps]
    assert all(b <= a + 1e-12 for a, b in zip(Js, Js[1:]))
