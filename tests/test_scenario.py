"""Tests for new-line scenarios (bkk.scenario)."""
import numpy as np
import pytest

from bkk.linemodel import ModelParams, OpenNetworkModel, build_line_network
from bkk.scenario import (
    METRO,
    TRAM,
    MinPlusEvaluator,
    NewLine,
    add_lines,
    find_segment,
    generate_candidates,
    greedy_select,
    line_cost,
)
from tests.test_linemodel import synthetic_feed


@pytest.fixture(scope="module")
def base():
    net = build_line_network(synthetic_feed())
    model = OpenNetworkModel(net, ModelParams())
    w = model.demand_prior()
    origins = np.arange(net.n_hubs)
    T = model.hub_travel_times(model.travel_graph(), origins)
    return net, model, w, origins, T


def _hub(net, stop):
    return net.stop_hub[stop]


def test_added_line_is_valid_subgenerator(base):
    net, *_ = base
    line = NewLine("x", TRAM, [_hub(net, "A"), _hub(net, "E")])
    aug = add_lines(net, [line])
    assert aug.n_segments == net.n_segments + 2
    rows = np.asarray(aug.succ.sum(axis=1)).ravel()
    np.testing.assert_allclose(rows + aug.seg_end, 1.0)
    m = OpenNetworkModel(aug, ModelParams())
    T = m.T.toarray()
    np.testing.assert_allclose(T.sum(axis=1) + m.exit, 0.0, atol=1e-14)


def test_min_plus_equals_dijkstra(base):
    net, model, w, origins, T = base
    ev = MinPlusEvaluator(T, origins, w)
    np.testing.assert_allclose(ev.efficiency(), model.efficiency(w), rtol=1e-12)
    line = NewLine("x", METRO, [_hub(net, "D"), _hub(net, "A"), _hub(net, "E")])
    aug = OpenNetworkModel(add_lines(net, [line]), ModelParams())
    exact = aug.efficiency(w, origins=origins)
    np.testing.assert_allclose(ev.efficiency(ev.updated(net, line)), exact, rtol=1e-12)
    assert exact > model.efficiency(w)


def test_through_running_keeps_passengers_on_board(base):
    net, *_ = base
    seg_db = find_segment(net, "BUS", _hub(net, "D"), _hub(net, "Bb"))
    seg_be = find_segment(net, "BUS", _hub(net, "Bb"), _hub(net, "E"))
    line = NewLine("thr", TRAM, [_hub(net, "B"), _hub(net, "C")], one_way=True,
                   feed_forward=[seg_db])
    aug = add_lines(net, [line])
    new = aug.n_segments - 1
    assert aug.succ[seg_db, new] == pytest.approx(1.0)
    assert aug.succ[seg_db, seg_be] == 0.0


def test_candidates_and_greedy(base):
    net, model, w, origins, T = base
    pool = np.flatnonzero(model.hub_active)
    spec = TRAM.__class__(**{**TRAM.__dict__, "min_len_m": 100.0, "spacing_m": 50.0})
    cands = generate_candidates(net, w, spec, endpoints=pool, station_pool=pool)
    assert cands and all(len(c.hubs) >= 2 for c in cands)
    picked = greedy_select(MinPlusEvaluator(T, origins, w), net, cands, k=2)
    gains = [p["marginal_gain"] for p in picked]
    assert all(g >= 0 for g in gains)
    assert picked[-1]["cumulative_gain"] == pytest.approx(sum(gains))
    assert all(p["cost"] == pytest.approx(line_cost(net, p["line"])) for p in picked)


def test_one_line_per_corridor(base):
    net, model, w, origins, T = base
    pool = np.flatnonzero(model.hub_active)
    spec = TRAM.__class__(**{**TRAM.__dict__, "min_len_m": 100.0, "spacing_m": 50.0})
    metro = METRO.__class__(**{**METRO.__dict__, "min_len_m": 100.0, "spacing_m": 50.0})
    cands = (generate_candidates(net, w, spec, pool, pool)
             + generate_candidates(net, w, metro, pool, pool))
    corridor = lambda c: frozenset((c.hubs[0], c.hubs[-1]))
    picked = greedy_select(MinPlusEvaluator(T, origins, w), net, cands, k=4, key=corridor)
    keys = [corridor(p["line"]) for p in picked]
    assert len(keys) == len(set(keys))
