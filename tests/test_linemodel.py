"""Tests for the line-aware open Markov model (bkk.linemodel) and audit fixes."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from bkk.constants import ROUTE_TYPES
from bkk.linemodel import (
    ModelParams,
    OpenNetworkModel,
    build_line_network,
    normalise_stop_name,
)
from bkk.resilience import ResilienceAnalyser
from bkk.simulate import TauLeap


def _hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def synthetic_feed():
    """Tram line A-B-C (both directions, 5-min headway) and a bus line D-B-E
    (10-min headway).  The bus stop at B is called 'B M' and lies 80 m from
    the tram stop, so both must merge into one interchange hub."""
    stops = pd.DataFrame({
        "stop_id":   ["A", "B", "Bb", "C", "D", "E"],
        "stop_name": ["A", "B", "B M", "C", "D", "E"],
        "stop_lat":  [47.50, 47.50, 47.5007, 47.50, 47.49, 47.51],
        "stop_lon":  [19.00, 19.01, 19.0100, 19.02, 19.01, 19.01],
        "parent_station": [np.nan] * 6,
    })
    routes = pd.DataFrame({"route_id": ["T", "BUS"], "route_type": ["0", "3"]})
    trips, st = [], []
    def add_trip(tid, rid, seq, t0, dt):
        trips.append({"trip_id": tid, "route_id": rid, "service_id": "x"})
        for k, s in enumerate(seq):
            st.append({"trip_id": tid, "stop_id": s, "stop_sequence": k + 1,
                       "arrival_time": _hms(t0 + k * dt), "departure_time": _hms(t0 + k * dt)})
    n = 0
    for t0 in np.arange(7 * 3600, 9 * 3600, 300):
        add_trip(f"t{n}", "T", ["A", "B", "C"], t0, 120); n += 1
        add_trip(f"t{n}", "T", ["C", "B", "A"], t0, 120); n += 1
    for t0 in np.arange(7 * 3600, 9 * 3600, 600):
        add_trip(f"b{n}", "BUS", ["D", "Bb", "E"], t0, 180); n += 1
    return SimpleNamespace(stops=stops, routes=routes, trips=pd.DataFrame(trips),
                           stop_times=pd.DataFrame(st))


@pytest.fixture(scope="module")
def net():
    return build_line_network(synthetic_feed())


@pytest.fixture(scope="module")
def model(net):
    return OpenNetworkModel(net, ModelParams(lam=1 / 300, mean_leg_m=1500, p_transfer=0.3))


# --------------------------------------------------------------------------- #
class TestNetwork:
    def test_suffix_normalisation(self):
        assert normalise_stop_name("Deák Ferenc tér M") == "Deák Ferenc tér"
        assert normalise_stop_name("Batthyány tér M+H") == "Batthyány tér"
        assert normalise_stop_name("Hild tér (Deák Ferenc tér M)") == "Hild tér (Deák Ferenc tér M)"

    def test_interchange_hub_merged(self, net):
        assert net.stop_hub["B"] == net.stop_hub["Bb"]
        assert net.n_hubs == 5

    def test_segments_and_frequencies(self, net):
        assert net.n_segments == 6                      # 4 tram + 2 bus
        tram = net.seg_mode == 0
        np.testing.assert_allclose(net.seg_freq[tram], 1 / 300)
        np.testing.assert_allclose(net.seg_freq[~tram], 1 / 600)
        np.testing.assert_allclose(net.seg_cv2, 0.0, atol=1e-12)   # regular headways

    def test_continuation_shares(self, net):
        rows = np.asarray(net.succ.sum(axis=1)).ravel()
        np.testing.assert_allclose(rows + net.seg_end, 1.0)
        assert np.all((net.seg_end == 0) | (net.seg_end == 1))


class TestGenerator:
    def test_subgenerator_structure(self, model):
        T = model.T.toarray()
        off = T - np.diag(np.diag(T))
        assert off.min() >= 0
        np.testing.assert_allclose(T.sum(axis=1) + model.exit, 0.0, atol=1e-14)
        assert np.all(model.exit >= 0) and model.exit.sum() > 0

    def test_headway_moment_matching(self, model, net):
        # regular service: E[W] = H/2, so the hazard is 2f, not f
        np.testing.assert_allclose(model.theta, 2 * net.seg_freq)

    def test_lambda_zero_is_frequency_share(self, net):
        m0 = OpenNetworkModel(net, ModelParams(lam=0.0))
        np.testing.assert_allclose(m0.accept, 1.0)
        np.testing.assert_allclose(m0.q_board, m0.theta)

    def test_acceptance_in_unit_interval(self, model):
        assert np.all((model.accept > 0) & (model.accept <= 1))


class TestSolution:
    def test_balance_and_little(self, model, net):
        lam = model.demand_prior() * 2.0
        L = model.steady_state(lam)
        assert np.all(L >= -1e-12)
        # L T + lambda = 0  <=>  total exit flow equals total arrival flow
        np.testing.assert_allclose(L @ model.exit, lam.sum(), rtol=1e-10)
        # Little's law
        np.testing.assert_allclose(L.sum(), lam.sum() * model.mean_journey_time(lam), rtol=1e-12)

    def test_scaling_is_linear(self, model):
        w = model.demand_prior()
        lam = model.scale_to_boardings(w, 10.0)
        np.testing.assert_allclose(model.flows(model.steady_state(lam))["boarding"].sum(), 10.0)
        np.testing.assert_allclose(model.steady_state(3 * lam), 3 * model.steady_state(lam))

    def test_transient_converges(self, model):
        lam = model.demand_prior()
        t = np.linspace(0, 20 * model.relaxation_time(), 5)
        N = model.transient_total(lam, t)
        assert N[0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(N) >= -1e-9)
        assert N[-1] == pytest.approx(model.steady_state(lam).sum(), rel=1e-6)

    def test_monte_carlo_matches_exact(self, model):
        lam = model.demand_prior()
        exact = model.mean_journey_time(lam)
        mc = model.sample_journeys(model.source_vector(lam), m=40_000,
                                   rng=np.random.default_rng(1))
        se = mc["journey_time"].std() / np.sqrt(40_000)
        assert abs(mc["journey_time"].mean() - exact) < 4 * se
        L = model.steady_state(lam)
        np.testing.assert_allclose(mc["occupancy_time"] * lam.sum(), L, rtol=0.05, atol=1e-3)


class TestResilienceMetrics:
    def test_efficiency_drops_under_closure(self, model, net):
        w = model.demand_prior()
        E0 = model.efficiency(w)
        hub_b = net.stop_hub["B"]
        assert model.efficiency(w, hub_b, "closure") < E0
        assert model.efficiency(w, hub_b, "failure") <= model.efficiency(w, hub_b, "closure")

    def test_closure_keeps_through_running(self, model, net):
        G = model.travel_graph(net.stop_hub["B"], "closure")
        from scipy.sparse.csgraph import dijkstra
        d = dijkstra(G, indices=[net.stop_hub["A"]])[0, : net.n_hubs]
        assert np.isfinite(d[net.stop_hub["C"]])        # ride A -> C through B
        G2 = model.travel_graph(net.stop_hub["B"], "failure")
        d2 = dijkstra(G2, indices=[net.stop_hub["A"]])[0, : net.n_hubs]
        assert np.isinf(d2[net.stop_hub["C"]])


# --------------------------------------------------------------------------- #
#  Regression tests for the audit fixes in the legacy modules                 #
# --------------------------------------------------------------------------- #
def test_current_route_types_supported():
    assert ROUTE_TYPES[11]["name"] == "trolleybus"
    assert ROUTE_TYPES[109]["name"] == "hev"


def test_departure_flux_uses_ctmc_stationary_law():
    # two-state cycle: in stationarity the flux 1->2 equals the flux 2->1
    flux = ResilienceAnalyser.departure_flux(np.array([0.5, 0.5]), np.array([1.0, 3.0]))
    np.testing.assert_allclose(flux, [0.75, 0.75])


def test_tau_leap_step_has_time_units():
    leaper = TauLeap(tau=5.0, epsilon=0.03)
    N = np.array([[10.0, 0.0]])
    v = np.array([0, 0]); i = np.array([0, 1]); q = np.array([0.2, 50.0])
    # only the occupied origin counts: tau = 0.03 / 0.2
    assert leaper._choose_tau(N, v, i, q) == pytest.approx(0.15)
    # doubling all rates halves the step (a time scale), unlike the old rule
    assert leaper._choose_tau(N, v, i, 2 * q) == pytest.approx(0.075)


def test_efficiency_is_exact_not_index_sampled():
    n = 300
    rows = np.arange(n); cols = (rows + 1) % n          # directed cycle
    P = csr_matrix((np.ones(n), (rows, cols)), shape=(n, n))
    E = ResilienceAnalyser._network_efficiency(P, np.full(n, 1 / n))
    expected = sum(1.0 / k for k in range(1, n)) / (n - 1)
    assert E == pytest.approx(expected, rel=1e-12)
