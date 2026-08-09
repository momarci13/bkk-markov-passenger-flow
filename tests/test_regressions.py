"""Regression tests for mathematical and multimodal integration semantics."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from bkk.generator import GeneratorBuilder, ModalGenerator
from bkk.gtfs import GTFSLoader
from bkk.network import ModalGraph, _parent_or_self
from bkk.simulate import GillespieSSA, TauLeap, allocate_modal_population


def _generator(route_type: int, stops: list[str]) -> ModalGenerator:
    n = len(stops)
    if n == 1:
        Q = csr_matrix((1, 1))
        P = csr_matrix((1, 1))
    else:
        P_dense = np.zeros((n, n))
        for i in range(n):
            P_dense[i, (i + 1) % n] = 1.0
        P = csr_matrix(P_dense)
        Q = csr_matrix(P_dense - np.eye(n))
    return ModalGenerator(route_type, str(route_type), n, np.array(stops), Q, P, 0.0)


def test_modal_allocation_conserves_at_interchanges():
    generators = {1: _generator(1, ["A", "B"]), 3: _generator(3, ["B", "C"])}
    global_counts = np.array([10.0, 20.0, 30.0])
    modal = allocate_modal_population(generators, global_counts, np.array(["A", "B", "C"]))
    assert modal[1].tolist() == [10.0, 10.0]
    assert modal[3].tolist() == [10.0, 30.0]
    assert sum(x.sum() for x in modal.values()) == pytest.approx(global_counts.sum())


def test_missing_parent_station_remains_its_own_stop():
    assert _parent_or_self("surface-stop", np.nan) == "surface-stop"
    assert _parent_or_self("surface-stop", "") == "surface-stop"
    assert _parent_or_self("platform", "station") == "station"


def test_generator_rate_is_per_passenger_inverse_time_hazard():
    graph = ModalGraph(
        route_type=3,
        name="bus",
        stop_ids=np.array(["A", "B"]),
        stop_idx={"A": 0, "B": 1},
        i_arr=np.array([0], dtype=np.int32),
        j_arr=np.array([1], dtype=np.int32),
        cost_arr=np.array([0.0]),
        mu_arr=np.array([1 / 600, 0.0]),
    )
    generator = GeneratorBuilder(lambda_v=0).build(graph)
    assert generator.Q[0, 1] == pytest.approx(1 / 600)
    # Ten independent passengers therefore have total propensity 10/600 s^-1.
    assert 10 * generator.Q[0, 1] == pytest.approx(1 / 60)


def test_stochastic_results_use_canonical_stop_coordinates():
    generators = {1: _generator(1, ["A", "B"]), 3: _generator(3, ["B", "C"])}
    modal = {1: np.array([4.0, 0.0]), 3: np.array([0.0, 7.0])}
    for result in (
        GillespieSSA(max_events=20, rng_seed=1).run(generators, modal, T=0.2),
        TauLeap(tau=0.05, epsilon=0, rng_seed=1).run(generators, modal, T=0.2, n_eval=5),
    ):
        assert result.metadata["stop_ids"].tolist() == ["A", "B", "C"]
        assert result.N_hist.shape[1] == 3
        assert result.N_hist[0].tolist() == [4.0, 0.0, 7.0]
        assert result.conservation_error() < 1e-12


def test_tauleap_returns_exact_requested_time_grid():
    generators = {1: _generator(1, ["A", "B"])}
    result = TauLeap(tau=10.0, epsilon=0).run(
        generators, {1: np.array([10.0, 0.0])}, T=4.0, n_eval=5
    )
    assert np.array_equal(result.t_eval, np.linspace(0, 4, 5))
    assert len(result.N_hist) == 5


def _calendar_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("stops.txt", "stop_id,stop_name,stop_lat,stop_lon\nA,A,0,0\nB,B,0,1\n")
        zf.writestr("routes.txt", "route_id,route_type\nR,3\n")
        zf.writestr(
            "trips.txt", "trip_id,route_id,service_id\nBASE,R,BASE\nADD,R,ADD\n"
        )
        zf.writestr(
            "stop_times.txt",
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "BASE,A,1,07:00:00,07:00:00\nBASE,B,2,07:05:00,07:05:00\n"
            "ADD,A,1,08:00:00,08:00:00\nADD,B,2,08:05:00,08:05:00\n",
        )
        zf.writestr(
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "BASE,1,1,1,1,1,0,0,20250101,20251231\n",
        )
        zf.writestr(
            "calendar_dates.txt",
            "service_id,date,exception_type\nBASE,20250106,2\nADD,20250106,1\n",
        )
    return buf.getvalue()


def test_exact_date_combines_base_calendar_and_exceptions():
    loader = GTFSLoader()
    loader._zip_bytes = _calendar_zip()
    feed = loader.parse(date_filter="20250106")
    assert set(feed.trips["trip_id"]) == {"ADD"}


def test_exact_date_without_service_fails_closed():
    loader = GTFSLoader()
    loader._zip_bytes = _calendar_zip()
    feed = loader.parse(date_filter="20260105")
    assert feed.trips.empty
