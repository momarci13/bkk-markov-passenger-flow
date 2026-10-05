"""Tests for the data-driven demand layers (bkk.demand_data)."""
import numpy as np
import pytest

from bkk.demand_data import (build_access, calibrate, calibrate_access, census_od_weights,
                             design_matrix, gravity_ipf, gravity_production, hub_catchments,
                             mode_response, solve_multipliers)
from bkk.linemodel import ModelParams, OpenNetworkModel, build_line_network
from bkk.scenario import MinPlusEvaluator

from .test_linemodel import synthetic_feed


@pytest.fixture(scope="module")
def model():
    net = build_line_network(synthetic_feed())
    return OpenNetworkModel(net, ModelParams(lam=1 / 300, mean_leg_m=1500, p_transfer=0.3))


def test_catchment_conserves_population():
    rng = np.random.default_rng(1)
    hubs = rng.uniform(0, 3000, (15, 2))
    cells = rng.uniform(-500, 3500, (4000, 2))
    pop = rng.exponential(10.0, 4000)
    P, U = hub_catchments(hubs, cells, pop, radius_m=600.0)
    assert P.sum() + U == pytest.approx(pop.sum())
    assert U > 0 and np.all(P >= 0)


def test_access_constant_shifts_population_to_class():
    rng = np.random.default_rng(2)
    hubs = rng.uniform(0, 2000, (20, 2))
    cls = np.where(np.arange(20) < 4, 1, 3)
    cells = rng.uniform(0, 2000, (3000, 2))
    pop = np.ones(3000)
    acc = build_access(hubs, cls, cells, pop, {1: 1000.0, 3: 500.0}, {1: 3, 3: 8})
    P0, U0 = acc.assign({1: 0.0})
    P1, U1 = acc.assign({1: 2.0})
    assert U0 == U1                                  # coverage does not depend on delta
    assert P0.sum() + U0 == pytest.approx(pop.sum())
    assert P1[cls == 1].sum() > P0[cls == 1].sum()


def test_census_blocks_and_ipf_marginals():
    rng = np.random.default_rng(3)
    n = 12
    T = rng.uniform(300, 3000, (n, n))
    o, a = rng.uniform(1, 5, n), rng.uniform(1, 5, n)
    district = np.repeat([0, 1, 2], 4)
    Q = rng.uniform(10, 50, (3, 3))
    W = census_od_weights(Q, district, o, a, T, beta=1e-3)
    for K in range(3):
        for L in range(3):
            assert W[np.ix_(district == K, district == L)].sum() == pytest.approx(Q[K, L])
    W2, info = gravity_ipf(o, a, T, beta=1e-3)
    assert np.allclose(W2.sum(axis=1), o, rtol=1e-6)
    assert np.allclose(W2.sum(axis=0), a * o.sum() / a.sum(), rtol=1e-6)
    W3 = gravity_production(o, a, T, beta=1e-3)
    assert np.allclose(W3.sum(axis=1), o)


def test_mode_response_is_linear_in_sources(model):
    net = model.net
    modes = sorted(set(net.seg_mode.tolist()))
    M = mode_response(model, modes)
    lam = np.where(model.hub_active, 1.0, 0.0) * 0.01
    F = model.flows(model.steady_state(lam))
    direct = [F["boarding_segment"][net.seg_mode == m].sum() for m in modes]
    assert np.allclose(lam @ M, direct)
    group = np.arange(net.n_hubs) % 2
    G = design_matrix(M, lam, group, 2)
    mu = np.array([0.5, 2.0])
    F2 = model.flows(model.steady_state(lam * mu[group]))
    direct2 = [F2["boarding_segment"][net.seg_mode == m].sum() for m in modes]
    assert np.allclose(G @ mu, direct2)


def test_nnls_recovers_planted_multipliers(model):
    net = model.net
    modes = sorted(set(net.seg_mode.tolist()))
    M = mode_response(model, modes)
    prior = np.where(model.hub_active, 1.0, 0.0) * 0.01
    group = (np.arange(net.n_hubs) % 2)
    G = design_matrix(M, prior, group, 2)
    if np.linalg.matrix_rank(G) < 2:
        pytest.skip("toy network does not identify two groups")
    mu_true = np.array([0.7, 1.6])
    target = G @ mu_true
    mu = solve_multipliers(G, target, rho=1e-10)
    assert np.allclose(mu, mu_true, rtol=1e-4)
    cal = calibrate(model, prior, group, 2, modes, G.sum(axis=1), rhos=np.array([1e-3]))
    assert np.allclose(cal.mu, 1.0, atol=1e-6)      # prior already fits -> mu = 1


def test_access_calibration_hits_attainable_target(model):
    net = model.net
    modes = sorted(set(net.seg_mode.tolist()))          # [0 tram, 3 bus]
    M = mode_response(model, modes)
    rng = np.random.default_rng(4)
    lat0 = np.radians(47.5)
    xy = np.column_stack([np.radians(net.hub_lon) * 6.371e6 * np.cos(lat0),
                          np.radians(net.hub_lat) * 6.371e6])
    cells = xy.mean(axis=0) + rng.uniform(-1500, 1500, (3000, 2))
    cls = np.array([0 if 0 in s else 3 for s in net.hub_modes()])
    acc = build_access(xy, cls, cells, np.ones(3000), {0: 1500.0, 3: 1500.0}, {0: 4, 3: 4},
                       model.hub_active)
    P, _ = acc.assign({0: 0.7})
    b = P @ M
    target = b / b.sum()
    delta, info = calibrate_access(acc, M, modes, target, free=(0,))
    assert delta[0] == pytest.approx(0.7, abs=1e-3)


def test_matrix_weights_match_vector_weights(model):
    net = model.net
    w = np.where(model.hub_active, net.hub_departures(), 0.0)
    w = w / w.sum()
    origins = np.arange(net.n_hubs)
    T = model.hub_travel_times(model.travel_graph(), origins)
    e_vec = MinPlusEvaluator(T, origins, w).efficiency()
    e_mat = MinPlusEvaluator(T, origins, np.outer(w, w)).efficiency()
    assert e_mat == pytest.approx(e_vec)
    assert model.efficiency(np.outer(w, w)) == pytest.approx(model.efficiency(w))
