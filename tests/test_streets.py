"""Tests for the GTFS-shape street graph (bkk.streets) and street-routed lines."""
import numpy as np
import pandas as pd
import pytest

from bkk.linemodel import ModelParams, OpenNetworkModel, build_line_network
from bkk.scenario import TRAM, MinPlusEvaluator, NewLine, add_lines, line_length_m
from bkk.streets import Router, build_street_graph, unproject
from tests.test_linemodel import synthetic_feed

X0, Y0 = 1_548_000.0, 5_281_000.0     # local metric coordinates near Budapest


def _shape(sid, pts):
    lat, lon = unproject(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return pd.DataFrame({"shape_id": sid, "shape_pt_sequence": range(len(pts)),
                         "shape_pt_lat": lat, "shape_pt_lon": lon})


@pytest.fixture(scope="module")
def l_graph():
    # tram along an L (east 1 km, then north 1 km); one bus on the diagonal
    shapes = pd.concat([
        _shape("T", [(X0, Y0), (X0 + 1000, Y0), (X0 + 1000, Y0 + 1000)]),
        _shape("B", [(X0, Y0), (X0 + 1000, Y0 + 1000)]),
    ])
    routes = pd.DataFrame({"shape_id": ["T", "B"], "route_id": ["t1", "b1"],
                           "route_type": ["0", "3"]})
    return build_street_graph(shapes, routes)


def test_edge_attributes(l_graph):
    A = l_graph.adj.tocoo()
    tram = np.asarray(l_graph.tram[A.row, A.col]).ravel() > 0.5
    nbus = np.asarray(l_graph.n_bus[A.row, A.col]).ravel()
    assert tram.any() and (~tram).any()
    assert set(np.round(nbus[~tram])) == {1.0}


def test_main_road_route_follows_streets(l_graph):
    lat, lon = unproject(np.array([X0, X0 + 1000]), np.array([Y0, Y0 + 1000]))
    any_bus = Router.build(l_graph, l_graph.adj, lat, lon)
    main = Router.build(l_graph, l_graph.main_roads(), lat, lon)
    d_any, _ = any_bus.paths_from(any_bus.hub_node[0])
    d_main, pred = main.paths_from(main.hub_node[0])
    assert d_any[any_bus.hub_node[1]] == pytest.approx(1414, rel=0.06)   # diagonal allowed
    assert d_main[main.hub_node[1]] == pytest.approx(2000, rel=0.06)     # L only
    path = main.extract(pred, main.hub_node[0], main.hub_node[1])
    xy = l_graph.xy[path]
    # every node of the main-road path lies on one of the two legs
    on_leg = (np.abs(xy[:, 1] - Y0) < 60) | (np.abs(xy[:, 0] - (X0 + 1000)) < 60)
    assert on_leg.all()


def test_street_lengths_drive_times_and_min_plus():
    net = build_line_network(synthetic_feed())
    model = OpenNetworkModel(net, ModelParams())
    w = model.demand_prior()
    origins = np.arange(net.n_hubs)
    T = model.hub_travel_times(model.travel_graph(), origins)
    hubs = [net.stop_hub["D"], net.stop_hub["A"], net.stop_hub["E"]]
    line = NewLine("s", TRAM, hubs, seg_len_m=[2500.0, 1800.0])
    assert line_length_m(net, line) == pytest.approx(4300.0)
    aug = add_lines(net, [line])
    new = aug.seg_route == "NEW:s"
    assert sorted(np.round(aug.seg_dist[new])) == [1800, 1800, 2500, 2500]
    exact = OpenNetworkModel(aug, ModelParams()).efficiency(w, origins=origins)
    ev = MinPlusEvaluator(T, origins, w)
    np.testing.assert_allclose(ev.efficiency(ev.updated(net, line)), exact, rtol=1e-12)
