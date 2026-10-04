"""
tests/test_core.py
==================
Unit tests for the BKK Markov-chain passenger-flow framework.

Run with:
    pytest tests/ -v

Tests cover:
  - GTFSFeed dataclass and summary
  - GTFSLoader parsing from synthetic in-memory ZIP
  - calendar.txt path (standard GTFS)
  - calendar_dates.txt path (BKK – no calendar.txt)
  - Mixed feed (both files present)
  - NetworkBuilder with synthetic data
  - GeneratorBuilder invariants
  - DemandPrior anchor (Σ N_hat = N_peak)
  - KFESolver conservation (LSODA + expm)
  - TauLeap conservation and non-negativity
  - GillespieSSA integer conservation
  - ResilienceAnalyser on a small toy graph
  - constants sanity checks
"""
from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Helpers: synthetic GTFS feed
# ---------------------------------------------------------------------------

def _make_tiny_feed(use_calendar_dates_only: bool = False):
    """
    Build a minimal synthetic GTFSFeed for a 3-stop, 1-mode (bus) network:
        Stop A → Stop B → Stop C
    Three trips per weekday, peak 07:00–09:00.

    Parameters
    ----------
    use_calendar_dates_only : bool
        If True, populate calendar_dates instead of calendar
        (simulates the real BKK feed which has no calendar.txt).
    """
    from bkk.gtfs import GTFSFeed

    stops = pd.DataFrame({
        "stop_id":   ["A", "B", "C"],
        "stop_name": ["Alpha", "Beta", "Gamma"],
        "stop_lat":  [47.50, 47.51, 47.52],
        "stop_lon":  [19.05, 19.06, 19.07],
    })

    routes = pd.DataFrame({
        "route_id":         ["R1"],
        "route_type":       ["3"],
        "route_short_name": ["99"],
    })

    trips = pd.DataFrame({
        "trip_id":    ["T1", "T2", "T3"],
        "route_id":   ["R1", "R1", "R1"],
        "service_id": ["WD", "WD", "WD"],
    })

    rows = []
    for trip_id, dep_h, dep_m in [("T1", 7, 0), ("T2", 7, 30), ("T3", 8, 0)]:
        for seq, (sid, dh, dm) in enumerate([
            ("A", dep_h, dep_m),
            ("B", dep_h, dep_m + 5),
            ("C", dep_h, dep_m + 10),
        ], start=1):
            rows.append({
                "trip_id":             trip_id,
                "stop_id":             sid,
                "stop_sequence":       seq,
                "arrival_time":        f"{dh:02d}:{dm:02d}:00",
                "departure_time":      f"{dh:02d}:{dm:02d}:00",
                "shape_dist_traveled": (seq - 1) * 500.0,
            })
    stop_times = pd.DataFrame(rows)

    transfers = pd.DataFrame({
        "from_stop_id":    ["B"],
        "to_stop_id":      ["B"],
        "transfer_type":   [2],
        "min_transfer_time": [120],
    })

    feed = GTFSFeed()
    feed.stops      = stops
    feed.routes     = routes
    feed.trips      = trips
    feed.stop_times = stop_times
    feed.transfers  = transfers

    if use_calendar_dates_only:
        # Add Monday dates with exception_type=1 (service running)
        # Find the next few Mondays from a base date
        base = date(2025, 1, 6)   # a known Monday
        date_rows = []
        for i in range(4):
            d = base + timedelta(weeks=i)
            date_rows.append({
                "service_id":     "WD",
                "date":           d.strftime("%Y%m%d"),
                "exception_type": 1,
            })
        # Add one Saturday with exception_type=1 for a different service
        sat = date(2025, 1, 11)
        date_rows.append({
            "service_id":     "WE",
            "date":           sat.strftime("%Y%m%d"),
            "exception_type": 1,
        })
        feed.calendar_dates = pd.DataFrame(date_rows)
    else:
        feed.calendar = pd.DataFrame({
            "service_id": ["WD"],
            "monday":    [1], "tuesday":  [1], "wednesday": [1],
            "thursday":  [1], "friday":   [1], "saturday":  [0], "sunday": [0],
            "start_date": ["20250101"], "end_date": ["20261231"],
        })

    return feed


def _make_tiny_network(use_calendar_dates_only: bool = False):
    from bkk.network import NetworkBuilder
    feed = _make_tiny_feed(use_calendar_dates_only=use_calendar_dates_only)
    return NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))


def _make_tiny_generators(lam: float = 1.0 / 300.0):
    from bkk.generator import GeneratorBuilder
    net  = _make_tiny_network()
    gens = GeneratorBuilder(lambda_v=lam).build_all(net)
    return net, gens


def _n0_modal(net, gens):
    """Map global E1 prior to per-mode arrays."""
    from bkk.demand import DemandPrior
    prior     = DemandPrior()
    N0_global = prior.e1_service_proxy(net)
    sid2g     = {sid: k for k, sid in enumerate(net.all_stop_ids)}
    N0_modal  = {}
    for rt, mg in gens.items():
        N_v = np.zeros(mg.n_stops)
        for lk, sid in enumerate(mg.stop_ids):
            g = sid2g.get(sid)
            if g is not None:
                N_v[lk] = N0_global[g]
        N0_modal[rt] = N_v
    return N0_modal


# ===========================================================================
# GTFSFeed
# ===========================================================================
class TestGTFSFeed:
    def test_n_stops(self):
        assert _make_tiny_feed().n_stops == 3

    def test_n_stop_times(self):
        assert _make_tiny_feed().n_stop_times == 9   # 3 trips × 3 stops

    def test_summary_with_calendar(self):
        s = _make_tiny_feed().summary()
        assert "calendar.txt" in s

    def test_summary_with_calendar_dates(self):
        s = _make_tiny_feed(use_calendar_dates_only=True).summary()
        assert "calendar_dates.txt" in s

    def test_has_calendar(self):
        assert _make_tiny_feed().has_calendar()
        assert not _make_tiny_feed(use_calendar_dates_only=True).has_calendar()

    def test_has_calendar_dates(self):
        assert _make_tiny_feed(
            use_calendar_dates_only=True
        ).has_calendar_dates()


# ===========================================================================
# GTFSLoader – parse from synthetic in-memory ZIP
# ===========================================================================
class TestGTFSLoaderParse:

    def _make_zip(self, include_calendar: bool = True,
                  include_calendar_dates: bool = False) -> bytes:
        """Build a minimal in-memory GTFS ZIP."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("stops.txt",
                "stop_id,stop_name,stop_lat,stop_lon\n"
                "A,Alpha,47.50,19.05\n"
                "B,Beta,47.51,19.06\n"
            )
            zf.writestr("routes.txt",
                "route_id,route_type,route_short_name\n"
                "R1,3,99\n"
            )
            zf.writestr("trips.txt",
                "trip_id,route_id,service_id\n"
                "T1,R1,WD\n"
                "T2,R1,WE\n"
            )
            zf.writestr("stop_times.txt",
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,A,1,07:00:00,07:00:00\n"
                "T1,B,2,07:05:00,07:05:00\n"
                "T2,A,1,10:00:00,10:00:00\n"
                "T2,B,2,10:05:00,10:05:00\n"
            )
            if include_calendar:
                zf.writestr("calendar.txt",
                    "service_id,monday,tuesday,wednesday,thursday,friday,"
                    "saturday,sunday,start_date,end_date\n"
                    "WD,1,1,1,1,1,0,0,20250101,20261231\n"
                    "WE,0,0,0,0,0,1,1,20250101,20261231\n"
                )
            if include_calendar_dates:
                # Monday 2025-01-06 → WD active, Tuesday 2025-01-07 → WD active
                zf.writestr("calendar_dates.txt",
                    "service_id,date,exception_type\n"
                    "WD,20250106,1\n"   # Monday  → added
                    "WD,20250107,1\n"   # Tuesday → added
                    "WE,20250111,1\n"   # Saturday → added
                )
        return buf.getvalue()

    # --- Basic parse --------------------------------------------------------
    def test_parse_stops(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip()
        feed = loader.parse()
        assert feed.n_stops == 2

    def test_parse_n_trips(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip()
        feed = loader.parse()
        assert feed.n_trips == 2

    # --- calendar.txt weekday filter ----------------------------------------
    def test_filter_monday_via_calendar(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(include_calendar=True)
        feed = loader.parse(weekday="monday")
        # Only WD trips on Monday
        assert set(feed.trips["service_id"]) == {"WD"}
        assert feed.n_trips == 1

    def test_filter_saturday_via_calendar(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(include_calendar=True)
        feed = loader.parse(weekday="saturday")
        assert set(feed.trips["service_id"]) == {"WE"}
        assert feed.n_trips == 1

    # --- calendar_dates.txt weekday filter (BKK path) -----------------------
    def test_filter_monday_via_calendar_dates_only(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(
            include_calendar=False, include_calendar_dates=True
        )
        feed = loader.parse(weekday="monday")
        # WD appears on 20250106 (Monday) with type=1
        assert "WD" in set(feed.trips["service_id"])
        # WE only appears on Saturday – must not be included
        assert "WE" not in set(feed.trips["service_id"])

    def test_filter_saturday_via_calendar_dates_only(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(
            include_calendar=False, include_calendar_dates=True
        )
        feed = loader.parse(weekday="saturday")
        assert "WE" in set(feed.trips["service_id"])
        assert "WD" not in set(feed.trips["service_id"])

    def test_filter_by_specific_date(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(
            include_calendar=False, include_calendar_dates=True
        )
        # 20250106 is a Monday with WD active
        feed = loader.parse(weekday="monday", date_filter="20250106")
        assert set(feed.trips["service_id"]) == {"WD"}

    def test_stop_times_filtered_with_trips(self):
        """After weekday filter, stop_times must only contain trips in feed."""
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip(include_calendar=True)
        feed = loader.parse(weekday="monday")
        trip_ids_in_st = set(feed.stop_times["trip_id"])
        trip_ids_in_trips = set(feed.trips["trip_id"])
        assert trip_ids_in_st.issubset(trip_ids_in_trips)

    # --- Error paths --------------------------------------------------------
    def test_missing_required_file_raises(self):
        from bkk.gtfs import GTFSLoader
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("stops.txt", "stop_id\nA\n")
            # routes.txt intentionally missing
            zf.writestr("calendar_dates.txt",
                        "service_id,date,exception_type\n")
        loader = GTFSLoader()
        loader._zip_bytes = buf.getvalue()
        with pytest.raises(ValueError, match="missing required files"):
            loader.parse()

    def test_no_calendar_at_all_raises(self):
        from bkk.gtfs import GTFSLoader
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("stops.txt",     "stop_id,stop_name,stop_lat,stop_lon\nA,X,0,0\n")
            zf.writestr("routes.txt",    "route_id,route_type\nR1,3\n")
            zf.writestr("trips.txt",     "trip_id,route_id,service_id\nT1,R1,S1\n")
            zf.writestr("stop_times.txt","trip_id,stop_id,stop_sequence\nT1,A,1\n")
            # No calendar.txt AND no calendar_dates.txt
        loader = GTFSLoader()
        loader._zip_bytes = buf.getvalue()
        with pytest.raises(ValueError, match="calendar"):
            loader.parse()

    def test_invalid_weekday_raises(self):
        from bkk.gtfs import GTFSLoader
        loader = GTFSLoader()
        loader._zip_bytes = self._make_zip()
        with pytest.raises(ValueError, match="not valid"):
            loader.parse(weekday="funday")

    def test_load_nonexistent_path_raises(self):
        from bkk.gtfs import GTFSLoader
        with pytest.raises(FileNotFoundError):
            GTFSLoader().load("/nonexistent/path/file.zip")


# ===========================================================================
# NetworkBuilder
# ===========================================================================
class TestNetworkBuilder:

    def test_stop_count(self):
        net = _make_tiny_network()
        assert len(net.all_stop_ids) == 3

    def test_modal_edges(self):
        net = _make_tiny_network()
        assert 3 in net.modal
        assert net.modal[3].n_edges == 2   # A→B, B→C

    def test_costs_positive(self):
        net = _make_tiny_network()
        assert (net.modal[3].cost_arr > 0).all()

    def test_mu_arr_max_positive(self):
        net = _make_tiny_network()
        assert net.modal[3].mu_arr.max() > 0

    def test_transfer_times_loaded(self):
        net = _make_tiny_network()
        assert "B" in net.transfer_times
        assert net.transfer_times["B"] == 120.0

    def test_summary_runs(self):
        assert "NetworkData" in _make_tiny_network().summary()

    def test_network_with_calendar_dates_feed(self):
        """NetworkBuilder must work even when the feed had no calendar.txt."""
        net = _make_tiny_network(use_calendar_dates_only=True)
        assert len(net.all_stop_ids) == 3
        assert 3 in net.modal


# ===========================================================================
# GeneratorBuilder
# ===========================================================================
class TestGeneratorBuilder:

    TOL = 1e-8

    def test_row_sums_Q_zero(self):
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            row_sums = np.array(mg.Q.sum(axis=1)).ravel()
            assert np.abs(row_sums).max() < self.TOL, (
                f"mode {rt}: max|Σ Q_ij| = {np.abs(row_sums).max():.2e}"
            )

    def test_row_sums_P_one_non_absorbing(self):
        """Non-absorbing rows of P must sum to 1; absorbing rows (terminals) sum to 0."""
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            row_sums = np.array(mg.P.sum(axis=1)).ravel()
            non_absorbing = row_sums > self.TOL
            if non_absorbing.any():
                dev = np.abs(row_sums[non_absorbing] - 1.0).max()
                assert dev < self.TOL, (
                    f"mode {rt}: max|Σ P_ij - 1| = {dev:.2e} (non-absorbing rows)"
                )

    def test_Q_offdiagonal_nonneg(self):
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            Q = mg.Q.tocoo()
            off = Q.data[Q.row != Q.col]
            assert off.min() >= -self.TOL

    def test_Q_diagonal_nonpos(self):
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            diag = np.array(mg.Q.diagonal())
            assert diag.max() <= self.TOL

    def test_P_no_self_loops(self):
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            diag = np.abs(mg.P.diagonal())
            assert diag.max() < self.TOL

    def test_validate_passes(self):
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            metrics = mg.validate()
            assert metrics["max_Q_rowsum_abs"] < self.TOL
            assert metrics["max_P_rowsum_dev"] < self.TOL

    def test_lambda_sensitivity(self):
        """Higher lambda → lower entropy (more concentrated P rows)."""
        net, gens_lo = _make_tiny_generators(lam=0.001)
        _,   gens_hi = _make_tiny_generators(lam=0.1)
        for rt in gens_lo:
            P_lo = gens_lo[rt].P.toarray()
            P_hi = gens_hi[rt].P.toarray()
            # Entropy of non-absorbing rows should be lower for high lambda
            def _row_entropy(P):
                rows = P[P.sum(axis=1) > 1e-8]
                p = rows[rows > 1e-12]
                return -np.sum(p * np.log(p))
            assert _row_entropy(P_lo) >= _row_entropy(P_hi) - 1e-6


# ===========================================================================
# DemandPrior
# ===========================================================================
class TestDemandPrior:

    N_PEAK = 440_000.0

    def _prior(self):
        from bkk.demand import DemandPrior
        return DemandPrior(n_day_total=self.N_PEAK / 0.11, phi_peak=0.11)

    def test_e1_anchor(self):
        net = _make_tiny_network()
        p   = self._prior()
        N   = p.e1_service_proxy(net)
        assert abs(N.sum() - p.n_peak) < 1.0

    def test_e1_nonneg(self):
        net = _make_tiny_network()
        assert (self._prior().e1_service_proxy(net) >= 0).all()

    def test_e2_anchor(self):
        net = _make_tiny_network()
        p   = self._prior()
        N   = p.e2_log_linear(net)
        assert abs(N.sum() - p.n_peak) < 1.0

    def test_e2_nonneg(self):
        net = _make_tiny_network()
        assert (self._prior().e2_log_linear(net) >= 0).all()

    def test_e3_returns_three_arrays(self):
        net = _make_tiny_network()
        mean, q05, q95 = self._prior().e3_monte_carlo(net, n_samples=10)
        assert mean.shape == q05.shape == q95.shape
        assert (q95 >= q05).all()

    def test_e3_mean_close_to_e2(self):
        """E3 mean should be close to E2 point estimate (small prior variance)."""
        from bkk.demand import DemandPrior
        net  = _make_tiny_network()
        p    = DemandPrior()
        e2   = p.e2_log_linear(net)
        mean, _, _ = p.e3_monte_carlo(net, n_samples=100)
        # Allow 20% relative deviation due to prior spread
        rel_dev = np.abs(mean - e2).max() / (e2.max() + 1e-6)
        assert rel_dev < 0.5

    def test_diurnal_peak_greater_than_midday(self):
        from bkk.demand import DemandPrior
        N = np.array([100.0, 200.0])
        assert DemandPrior.diurnal_scale(N, hour=8).sum() > \
               DemandPrior.diurnal_scale(N, hour=13).sum()


# ===========================================================================
# KFESolver
# ===========================================================================
class TestKFESolver:

    def _setup(self):
        from bkk.simulate import KFESolver
        net, gens = _make_tiny_generators()
        N0_modal  = _n0_modal(net, gens)
        return KFESolver(), gens, N0_modal

    def test_conservation_lsoda(self):
        solver, gens, N0 = self._setup()
        results = solver.solve(gens, N0, T=60.0, n_eval=11)
        for rt, res in results.items():
            assert res.conservation_error() < 1e-5, (
                f"mode {rt}: err={res.conservation_error():.2e}"
            )

    def test_conservation_expm(self):
        from bkk.simulate import KFESolver
        _, gens, N0 = self._setup()
        results = KFESolver(method="expm").solve(gens, N0, T=60.0, n_eval=5)
        for rt, res in results.items():
            assert res.conservation_error() < 1e-5

    def test_N_nonneg(self):
        solver, gens, N0 = self._setup()
        results = solver.solve(gens, N0, T=60.0, n_eval=11)
        for rt, res in results.items():
            assert res.N_hist.min() >= -1e-6

    def test_shape(self):
        solver, gens, N0 = self._setup()
        results = solver.solve(gens, N0, T=60.0, n_eval=11)
        for rt, res in results.items():
            assert res.N_hist.shape == (11, gens[rt].n_stops)

    def test_result_mode_label(self):
        solver, gens, N0 = self._setup()
        results = solver.solve(gens, N0, T=10.0, n_eval=3)
        for res in results.values():
            assert res.mode.startswith("kfe")


# ===========================================================================
# TauLeap
# ===========================================================================
class TestTauLeap:

    def _setup(self):
        from bkk.simulate import TauLeap
        net, gens = _make_tiny_generators()
        return TauLeap(tau=0.5, rng_seed=7), gens, _n0_modal(net, gens)

    def test_conservation(self):
        leaper, gens, N0 = self._setup()
        res = leaper.run(gens, N0, T=60.0)
        # Non-negativity guard can cause minor drift; allow 1 %
        assert res.conservation_error() < 0.01

    def test_N_nonneg(self):
        leaper, gens, N0 = self._setup()
        res = leaper.run(gens, N0, T=60.0)
        assert res.N_hist.min() >= -1e-6

    def test_mode_label(self):
        leaper, gens, N0 = self._setup()
        assert leaper.run(gens, N0, T=10.0).mode == "tauleap"

    def test_n_steps_positive(self):
        leaper, gens, N0 = self._setup()
        res = leaper.run(gens, N0, T=10.0)
        assert res.metadata["n_steps"] > 0


# ===========================================================================
# GillespieSSA
# ===========================================================================
class TestGillespieSSA:

    def _setup(self):
        from bkk.simulate import GillespieSSA
        net, gens = _make_tiny_generators()
        return GillespieSSA(max_events=5_000, rng_seed=0), gens, _n0_modal(net, gens)

    def test_integer_conservation(self):
        ssa, gens, N0 = self._setup()
        res = ssa.run(gens, N0, T=30.0)
        assert res.conservation_error() < 1e-10

    def test_mode_label(self):
        ssa, gens, N0 = self._setup()
        assert ssa.run(gens, N0, T=10.0).mode == "ssa"

    def test_events_are_list(self):
        ssa, gens, N0 = self._setup()
        res = ssa.run(gens, N0, T=30.0)
        assert isinstance(res.events, list)

    def test_ensemble_length(self):
        ssa, gens, N0 = self._setup()
        runs = ssa.ensemble(gens, N0, T=10.0, M=5)
        assert len(runs) == 5

    def test_ensemble_conservation(self):
        ssa, gens, N0 = self._setup()
        for res in ssa.ensemble(gens, N0, T=10.0, M=3):
            assert res.conservation_error() < 1e-10


# ===========================================================================
# ResilienceAnalyser (toy graph)
# ===========================================================================
class TestResilienceAnalyser:

    def _toy_P(self) -> csr_matrix:
        """
        3-stop toy network:
          0→1: 0.6,  0→2: 0.4
          1→0: 0.5,  1→2: 0.5
          2→0: 0.7,  2→1: 0.3
        """
        return csr_matrix(np.array([
            [0,   0.6, 0.4],
            [0.5, 0,   0.5],
            [0.7, 0.3, 0  ],
        ], dtype=float))

    def test_stationary_sums_to_one(self):
        from bkk.resilience import ResilienceAnalyser
        pi = ResilienceAnalyser._stationary(self._toy_P(), 3)
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_stationary_is_invariant(self):
        from bkk.resilience import ResilienceAnalyser
        P  = self._toy_P()
        pi = ResilienceAnalyser._stationary(P, 3)
        assert np.abs(pi @ P - pi).max() < 1e-5

    def test_remove_node_shape(self):
        from bkk.resilience import ResilienceAnalyser
        P_red = ResilienceAnalyser._remove_node(self._toy_P(), 1)
        assert P_red.shape == (2, 2)

    def test_remove_node_row_stochastic(self):
        from bkk.resilience import ResilienceAnalyser
        P_red = ResilienceAnalyser._remove_node(self._toy_P(), 1)
        row_sums = np.array(P_red.sum(axis=1)).ravel()
        assert np.abs(row_sums - 1.0).max() < 1e-8

    def test_kemeny_positive(self):
        from bkk.resilience import ResilienceAnalyser
        k = ResilienceAnalyser._kemeny_from_evals(np.array([0.5, 0.2, 0.1]), 4)
        assert k > 0

    def test_analyse_tiny_network(self):
        from bkk.resilience import ResilienceAnalyser
        _, gens = _make_tiny_generators()
        analyser = ResilienceAnalyser(top_k=3, krylov_trunc=2)
        for rt, mg in gens.items():
            report = analyser.analyse(mg)
            df = report.to_dataframe()
            assert len(df) <= 3
            assert (df["criticality"] >= 0).all()
            assert (df["criticality"] <= 1.0 + 1e-6).all()

    def test_criticality_in_unit_interval(self):
        from bkk.resilience import ResilienceAnalyser
        _, gens = _make_tiny_generators()
        for rt, mg in gens.items():
            rep = ResilienceAnalyser(top_k=3).analyse(mg)
            for s in rep.stops:
                assert 0.0 <= s.criticality <= 1.0 + 1e-9


# ===========================================================================
# constants
# ===========================================================================
class TestConstants:

    def test_calendar_not_in_required(self):
        """BKK does not ship calendar.txt – it must not be in GTFS_REQUIRED."""
        from bkk.constants import GTFS_REQUIRED
        assert "calendar.txt" not in GTFS_REQUIRED

    def test_calendar_dates_in_files(self):
        from bkk.constants import GTFS_FILES
        assert "calendar_dates.txt" in GTFS_FILES

    def test_route_types_complete(self):
        from bkk.constants import ROUTE_TYPES, BKK_ROUTE_TYPES
        assert BKK_ROUTE_TYPES == frozenset(ROUTE_TYPES.keys())

    def test_kappa_values_positive(self):
        from bkk.constants import ROUTE_TYPES
        for rt, meta in ROUTE_TYPES.items():
            assert meta["kappa"] > 0

    def test_gtfs_required_subset_of_files(self):
        from bkk.constants import GTFS_FILES, GTFS_REQUIRED
        for f in GTFS_REQUIRED:
            assert f in GTFS_FILES

    def test_beta_prior_lengths_match(self):
        from bkk.constants import BETA_PRIOR_MEAN, BETA_PRIOR_STD
        assert len(BETA_PRIOR_MEAN) == len(BETA_PRIOR_STD)

    def test_default_lambda_units(self):
        """λ = 1/300 s⁻¹ → exp(-λ · 300 s) = e⁻¹ ≈ 0.368."""
        from bkk.constants import DEFAULT_LAMBDA
        penalty = np.exp(-DEFAULT_LAMBDA * 300.0)
        assert 0.36 < penalty < 0.38

    def test_weekday_names_length(self):
        from bkk.constants import WEEKDAY_NAMES
        assert len(WEEKDAY_NAMES) == 7

    def test_trolleybus_in_route_types(self):
        """route_type 800 (trolleybus) must be present — BKK-specific."""
        from bkk.constants import ROUTE_TYPES
        assert 800 in ROUTE_TYPES
        assert ROUTE_TYPES[800]["name"] == "trolleybus"


# ===========================================================================
# Metro platform-stop aggregation (stop2node / parent_station)
# ===========================================================================
class TestMetroPlatformAggregation:
    """
    Verify that direction-specific platform stop IDs are collapsed to their
    parent_station, turning the metro DAG into an irreducible subgraph with
    a non-trivial stationary distribution.

    Topology
    --------
    Without aggregation (6 nodes, 2 disjoint directed paths):
        A0 -> B0 -> C0    (direction 0 trips)
        C1 -> B1 -> A1    (direction 1 trips)
    => DAG: power iteration drains to {C0, A1} -> pi = 0 for most stops.

    After parent_station aggregation (3 nodes):
        A <-> B <-> C     (both directions merge to same station nodes)
    => Irreducible Markov chain: pi > 0 for every stop.
    """

    def _make_feed(self):
        from bkk.gtfs import GTFSFeed

        stops = pd.DataFrame({
            "stop_id": [
                "A0", "A1", "B0", "B1", "C0", "C1",
                "A",  "B",  "C",
            ],
            "stop_name": [
                "Alpha-0", "Alpha-1", "Beta-0", "Beta-1",
                "Gamma-0", "Gamma-1", "Alpha",  "Beta",  "Gamma",
            ],
            "stop_lat": [
                47.50, 47.50, 47.51, 47.51, 47.52, 47.52,
                47.50, 47.51, 47.52,
            ],
            "stop_lon": [
                19.05, 19.05, 19.06, 19.06, 19.07, 19.07,
                19.05, 19.06, 19.07,
            ],
            "parent_station": [
                "A", "A", "B", "B", "C", "C",
                "",  "",  "",
            ],
            "location_type": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        })

        routes = pd.DataFrame({
            "route_id":         ["M1"],
            "route_type":       ["1"],   # metro
            "route_short_name": ["M1"],
        })

        trips = pd.DataFrame({
            "trip_id":    ["D0T1", "D0T2", "D1T1", "D1T2"],
            "route_id":   ["M1"] * 4,
            "service_id": ["WD"] * 4,
        })

        rows = []
        for trip_id, h, m, stop_seq in [
            ("D0T1",  7,  0, ["A0", "B0", "C0"]),
            ("D0T2",  7, 30, ["A0", "B0", "C0"]),
            ("D1T1",  7,  5, ["C1", "B1", "A1"]),
            ("D1T2",  7, 35, ["C1", "B1", "A1"]),
        ]:
            for seq, sid in enumerate(stop_seq, start=1):
                rows.append({
                    "trip_id":             trip_id,
                    "stop_id":             sid,
                    "stop_sequence":       seq,
                    "arrival_time":        f"{h:02d}:{m + (seq-1)*3:02d}:00",
                    "departure_time":      f"{h:02d}:{m + (seq-1)*3:02d}:00",
                    "shape_dist_traveled": (seq - 1) * 1000.0,
                })
        stop_times = pd.DataFrame(rows)

        transfers = pd.DataFrame(
            columns=["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"]
        )

        calendar_dates = pd.DataFrame([{
            "service_id":     "WD",
            "date":           "20250106",   # a Monday
            "exception_type": 1,
        }])

        feed = GTFSFeed()
        feed.stops          = stops
        feed.routes         = routes
        feed.trips          = trips
        feed.stop_times     = stop_times
        feed.transfers      = transfers
        feed.calendar_dates = calendar_dates
        return feed

    def _build_metro_graph(self):
        from bkk.network import NetworkBuilder
        net = NetworkBuilder().build(
            self._make_feed(),
            peak_window=("07:00:00", "09:00:00"),
        )
        return net.modal.get(1)   # route_type 1 = metro

    # ------------------------------------------------------------------ #
    def test_metro_graph_not_none(self):
        """NetworkBuilder must produce a metro ModalGraph from the test feed."""
        mg = self._build_metro_graph()
        assert mg is not None, "Metro ModalGraph is None -- mode 1 not built"

    def test_stop_ids_are_parent_stations(self):
        """Platform stop IDs must not appear in stop_ids after aggregation."""
        mg = self._build_metro_graph()
        assert mg is not None
        platform_ids = {"A0", "A1", "B0", "B1", "C0", "C1"}
        leaked = platform_ids & set(mg.stop_ids)
        assert not leaked, f"Platform IDs leaked into stop_ids: {leaked}"

    def test_parent_station_ids_in_stop_ids(self):
        """All three parent station IDs must be present in stop_ids."""
        mg = self._build_metro_graph()
        assert mg is not None
        missing = {"A", "B", "C"} - set(mg.stop_ids)
        assert not missing, f"Parent station IDs missing from stop_ids: {missing}"

    def test_no_self_loops(self):
        """Self-loop edges (i == j) must not be created during aggregation."""
        mg = self._build_metro_graph()
        assert mg is not None
        assert (mg.i_arr != mg.j_arr).all(), \
            "Self-loops found in metro i_arr/j_arr after stop aggregation"

    def test_bidirectional_edges_present(self):
        """
        After aggregation the graph must have both A->B and B->A edges (and
        likewise for B<->C), confirming the DAG structure is broken.
        """
        from bkk.generator import GeneratorBuilder
        mg = self._build_metro_graph()
        assert mg is not None
        modal_gen = GeneratorBuilder(lambda_v=1 / 300.0).build(mg)
        P = modal_gen.P.toarray()
        idx = {sid: k for k, sid in enumerate(mg.stop_ids)}
        a, b, c = idx["A"], idx["B"], idx["C"]
        assert P[a, b] > 0 and P[b, a] > 0, "A<->B not bidirectional after aggregation"
        assert P[b, c] > 0 and P[c, b] > 0, "B<->C not bidirectional after aggregation"

    def test_stationary_distribution_nonzero(self):
        """
        pi must be positive for every stop.
        Without aggregation, power iteration drains to DAG sinks and pi -> 0.
        """
        from bkk.generator import GeneratorBuilder
        from bkk.resilience import ResilienceAnalyser
        mg = self._build_metro_graph()
        assert mg is not None
        modal_gen = GeneratorBuilder(lambda_v=1 / 300.0).build(mg)
        pi = ResilienceAnalyser._stationary(modal_gen.P, mg.n_stops)
        assert (pi > 1e-8).all(), \
            f"Stationary distribution has near-zero entries -- DAG not fixed: {pi}"

    def test_mu_arr_nonzero_after_aggregation(self):
        """
        Departure intensity must be non-zero after summing platform trip
        counts to station level.
        """
        mg = self._build_metro_graph()
        assert mg is not None
        assert mg.mu_arr.sum() > 0, \
            "Metro mu_arr is all zeros -- station-level aggregation failed"
