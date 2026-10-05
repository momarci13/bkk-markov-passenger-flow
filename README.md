# BKK Markov-Chain Passenger-Flow Framework

Multi-modal stochastic passenger-flow model for the **BKK Budapest** public transit network,
calibrated from the publicly available GTFS schedule.

The framework turns schedules into modal continuous-time Markov chains, supports
deterministic and stochastic passenger-flow simulation, and measures network resilience
under node removal. It is designed as a transparent research framework rather than a
passenger-count forecasting product.

**Data source:** [bkk.hu/bkk-partnerek/egyeb/gtfs-menetrendi-adatok-programozoknak/](https://bkk.hu/bkk-partnerek/egyeb/gtfs-menetrendi-adatok-programozoknak/)
**Download URL:** `https://bkk.hu/gtfs/budapest_gtfs.zip`
**Licence:** CC0-1.0

---

## Version 3 — line-aware open Markov model and TDK paper

A mathematical audit (`AUDIT.md`) found structural errors in the v2 stop-hop chain
(passengers alighting at every stop, no in-vehicle time, a 2× waiting hazard,
stock/flow confusion, mixed stationary laws) and two data defects (route types 11/109
dropped, weekday-union calendar inflating frequencies 2.4×). Version 3 adds
`bkk/linemodel.py`:

* states = hub waiting ∪ line-segment riding; continuation shares from GTFS trips
* boarding hazard θ = 2f/(1+CV²) (renewal mean wait), boarding choice by minimum relative
  entropy realised exactly by Poisson thinning
* open network with Poisson inflow → occupancies are **independent Poisson**, journey time is
  phase-type; solved exactly with one sparse linear solve (simulation only for verification)
* demand-weighted efficiency with exact Dijkstra for station-closure / node-failure analysis
* `bkk/scenario.py`: new-line scenarios — exact min-plus update of all shortest times for a
  candidate line, lazy greedy selection by efficiency gain per construction cost, through-running
  into existing lines (used for the planned M5 benchmark)
* `bkk/streets.py`: street graph from GTFS `shapes.txt` (40 m grid); new trams are routed by
  shortest path on main roads only (tram track, trolleybus, or ≥ 3 bus routes) and cross the
  Danube only on today's tram bridges; metro stays a straight tunnel
* `bkk/demand_data.py`: population-based demand. Meta HRSL 30 m residents are assigned to
  hubs by a logit access-hub choice, `ω_ch ∝ exp(−d_ch/ℓ + δ_k(h))`, with class radii
  (metro/HÉV 1000 m, tram 600 m, bus 500 m). The access constants δ are calibrated to BKK
  trips by mode through the linear mode response `M = (−T)⁻¹B`, i.e. `b_m(λ) = λᵀ M e_m`.
  The script also runs a district NNLS/BVLS identifiability check. OD weights come from a
  production-constrained gravity model, `W_od = λ_o a_d e^{−βt_od} / Σ_d' …`; the main
  case uses β = 0 and the calibrated β is a robustness check. Census district OD and
  car-ownership tables are supported as manual CSV exports (`data/external/manual/`).
  Select the demand variant with `TDK_DEMAND=population|population_gravity|departures`.

The Hungarian TDK paper is `tdk/tdk_dolgozat.pdf`. Every number in it is reproduced by:

```bash
# official BKK feed via the MobilityData mirror (version 2572.20260604 was used)
curl -L -o data/budapest_gtfs.zip \
  "https://storage.googleapis.com/storage/v1/b/mdb-latest/o/hu-budapest-budapesti-kozlekedesi-kozpont-bkk-gtfs-990.zip?alt=media"
python scripts/tdk_demand_data.py                # HRSL window (S3), access-choice calibration
python scripts/tdk_analysis.py --date 20260609   # ~15 min, writes data/results/
python scripts/tdk_scenarios.py                  # new metro/tram lines + planned M5 benchmark
python scripts/tdk_plans.py                      # current Budapest plans (Bajcsy tram, Budai fonódó II,
                                                 # Budafoki út tram, M5) + best lines after them
python scripts/tdk_plans_demand.py               # 2030 South-Buda housing demand scenario
python scripts/tdk_figures.py                    # Budapest maps (EOV), downloads geoBoundaries
python scripts/tdk_tex_numbers.py                # LaTeX macros + table
cd tdk && pdflatex tdk_dolgozat.tex && pdflatex tdk_dolgozat.tex
```

`bp_markov_revised.*` and `prezi_revised.*` describe the superseded v2 model and are kept
for provenance only; see `AUDIT.md` for what is wrong in them.

---

## GTFS File Inventory

The BKK ZIP contains these standard GTFS Schedule files:

| File | Contents | Model role |
|------|----------|-----------|
| `agency.txt` | BKK operator metadata | Identification |
| `stops.txt` | ~6 200 stops with lat/lon | Vertex set S, n ≈ 5 551 active |
| `routes.txt` | 368 routes with `route_type` | Mode classification v ∈ V |
| `trips.txt` | 178 083 trips | Trip → route → calendar |
| `stop_times.txt` | 3.6 M departure/arrival records | Travel times τ^v_{ij}, intensities μ^v_i |
| `calendar.txt` | Weekday service flags | Weekday filter |
| `calendar_dates.txt` | Exception dates | Added/removed service days |
| `shapes.txt` | 685 179 route waypoints | Distance d^v_{ij} via shape_dist_traveled |
| `transfers.txt` | Transfer times per stop | Penalty δ · t_{tf}(i) |
| `feed_info.txt` | Feed version/dates | Metadata |

**BKK `route_type` codes** (extended GTFS):

| `route_type` | Mode | Hungarian name |
|---|---|---|
| 0 | Tram | villamos |
| 1 | Metro | metró |
| 2 | Suburban railway | HÉV |
| 3 | Bus | busz |
| 4 | Ferry | hajó |
| 800 / 11 | Trolleybus | trolibusz |
| 109 | Suburban railway (current feeds) | HÉV |

---

## Installation

```bash
pip install -e ".[all]"
```

Or minimal (no geo / visualisation):

```bash
pip install -e .
```

Run the complete public-data pipeline:

```bash
bkk-download --cache-dir data/
python scripts/run_pipeline.py --zip data/budapest_gtfs.zip
```

---

## Quick Start

### Download and parse GTFS

```python
from bkk import GTFSLoader

loader = GTFSLoader(cache_dir="data/")
loader.download()                       # saves data/budapest_gtfs.zip
feed = loader.parse(weekday="monday")   # filter to Monday service
print(feed.summary())
```

Or from a local ZIP:

```python
feed = GTFSLoader().load("data/budapest_gtfs.zip").parse(weekday="monday")
```

### Build modal subgraphs and cost matrices

```python
from bkk import NetworkBuilder

net = NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))
print(net.summary())
# Access a specific mode (route_type 3 = bus)
bus_graph = net.modal[3]
print(f"Bus: {bus_graph.n_stops} stops, {bus_graph.n_edges} edges")
print(f"     fill = {bus_graph.n_edges / bus_graph.n_stops**2 * 100:.2f}%")
```

### Build CTMC generator matrices

```python
from bkk import GeneratorBuilder

gen_builder = GeneratorBuilder(lambda_v=1/300)   # λ^v = 1/300 s^{-1}
generators  = gen_builder.build_all(net)

for rt, mg in generators.items():
    metrics = mg.validate()
    print(f"mode {rt}: max|ΣQ_ij|={metrics['max_Q_rowsum_abs']:.1e}  "
          f"max|ΣP_ij-1|={metrics['max_P_rowsum_dev']:.1e}")
```

### Estimate initial passenger counts

```python
from bkk import DemandPrior
import numpy as np

prior     = DemandPrior()           # N_day=4e6, phi_peak=0.11
N0_global = prior.e1_service_proxy(net)   # E1: service proxy (GTFS only)
print(f"Peak-hour estimate: {N0_global.sum():.0f} passengers")
print(f"Top stop: {net.all_stop_ids[np.argmax(N0_global)]}")
```

Or with external population and POI data (E2):

```python
pop_i = np.zeros(len(net.all_stop_ids))   # replace with WorldPop raster query
poi_i = np.zeros(len(net.all_stop_ids))   # replace with OSM Overpass query
N0_e2 = prior.e2_log_linear(net, pop_i=pop_i, poi_i=poi_i)

# Monte-Carlo uncertainty propagation (E3)
mean_N, q05, q95 = prior.e3_monte_carlo(net, n_samples=200)
```

### Solve the Kolmogorov Forward Equation (KFE)

```python
from bkk import KFESolver, allocate_modal_population

# Split each stop's population only among modes that serve that stop.
N0_modal = allocate_modal_population(
    generators, N0_global, net.all_stop_ids
)

solver  = KFESolver(method="lsoda")
results = solver.solve(generators, N0_modal, T=900.0, n_eval=61)

for rt, res in results.items():
    print(f"mode {rt}: conservation_err={res.conservation_error():.2e}")
    print(f"         N_final_sum={res.N_final.sum():.1f}")
```

### τ-leaping stochastic simulation

```python
from bkk import TauLeap

leaper = TauLeap(tau=1.0, rng_seed=42)
result = leaper.run(generators, N0_modal, T=900.0)
print(f"Steps: {result.metadata['n_steps']}")
print(f"Conservation error: {result.conservation_error():.2e}")
```

### Exact Gillespie SSA (small networks / short windows)

```python
from bkk import GillespieSSA

ssa  = GillespieSSA(max_events=50_000)
res  = ssa.run(generators, N0_modal, T=60.0)   # 1-minute window
print(f"Events fired: {res.metadata['n_events']}")

# Ensemble of M realisations
ensemble = ssa.ensemble(generators, N0_modal, T=900.0, M=40)
```

### Network resilience analysis

```python
from bkk import ResilienceAnalyser

analyser = ResilienceAnalyser(top_k=100)
reports  = analyser.analyse_all(generators, N0_modal)

for rt, rep in reports.items():
    df = rep.to_dataframe()
    print(f"\nMode {rt} – top-10 critical stops:")
    print(df[["stop_id", "importance", "delta_kemeny",
              "delta_eff", "criticality"]].head(10).to_string())
```

---

## Command-Line Interface

```bash
# Download GTFS
bkk-download --cache-dir data/

# Build network (parse + generators + validation)
bkk-build --zip data/budapest_gtfs.zip --lambda-v 0.00333

# Simulate (KFE, SSA, or tau-leap)
bkk-simulate --zip data/budapest_gtfs.zip --mode kfe --horizon 900
bkk-simulate --zip data/budapest_gtfs.zip --mode tauleap --tau 1.0
bkk-simulate --zip data/budapest_gtfs.zip --mode ssa --ensemble 10

# Resilience analysis → data/resilience_scores.csv
bkk-resilience --zip data/budapest_gtfs.zip --top-k 100
```

Or run the full pipeline:

```bash
python scripts/run_pipeline.py --zip data/budapest_gtfs.zip
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite covers:
- GTFS parsing (synthetic ZIP)
- Generator invariants: `max|ΣQ_ij| < 1e-8`, `max|ΣP_ij-1| < 1e-8`, no negative off-diagonals, no self-loops
- Demand prior anchoring: `sum(N_hat) == N_peak`
- KFE conservation: `|N(T) - N(0)| / N(0) < 1e-5`
- τ-leap conservation and non-negativity
- SSA exact integer conservation
- Resilience metrics (stationary distribution, node removal, Kemeny)

---

## Mathematical Summary

| Symbol | Definition | Units |
|--------|-----------|-------|
| C^v_{ij} | κ^v · τ̄^v_{ij} + γ · d^v_{ij} + δ · I_{tf} · t_{tf}(i) | s |
| λ^v | Entropy–cost multiplier | s⁻¹ |
| π^v_{j\|i} | exp(-λ^v C^v_{ij}) / Σ_k exp(-λ^v C^v_{ik}) | dimensionless |
| P^v_{ij} | π^v_{j\|i}, j≠i; 0 for j=i (embedded jump chain) | dimensionless |
| μ^v_i | scheduled departures from stop i per second | s⁻¹ |
| q^v_{ij} | μ^v_i · π^v_{j\|i} | s⁻¹ |
| Q^v_{ii} | -Σ_{j≠i} q^v_{ij} | s⁻¹ |

KFE: d**N**^v/dt = **N**^v **Q**^v, solution **N**^v(t) = **N**^v(0) exp(**Q**^v t)

---

## Project Structure

```
bkk_framework/
├── bkk/
│   ├── __init__.py      Public API
│   ├── constants.py     BKK-specific constants, GTFS route_type map
│   ├── gtfs.py          GTFSLoader: download + parse budapest_gtfs.zip
│   ├── network.py       NetworkBuilder: modal subgraphs, cost matrices, μ^v_i
│   ├── generator.py     GeneratorBuilder: CTMC Q^v and DTMC P^v (CSR)
│   ├── demand.py        DemandPrior: E1/E2/E3 estimators for N_i(0)
│   ├── simulate.py      KFESolver, GillespieSSA, TauLeap
│   ├── resilience.py    ResilienceAnalyser: Kemeny/gap/efficiency
│   ├── linemodel.py     v3 line-aware open Markov network (exact solution)
│   ├── scenario.py      new-line scenarios (min-plus screening, greedy selection)
│   ├── streets.py       GTFS-shape street graph and main-road router
│   ├── demand_data.py   population catchments, access choice, gravity OD, NNLS/BVLS
│   └── cli.py           CLI entry points
├── tests/
│   ├── test_core.py         Core mathematical and interface tests
│   ├── test_regressions.py  Regression tests for previously identified edge cases
│   └── test_linemodel.py    v3 model + audit-fix regression tests
├── scripts/
│   ├── run_pipeline.py  End-to-end pipeline script (legacy v2)
│   ├── tdk_analysis.py  v3 analysis behind the TDK paper
│   ├── tdk_scenarios.py new metro/tram lines and the M5 benchmark
│   ├── tdk_figures.py   maps and figures
│   ├── budapest_basemap.py  vector basemap (districts, Danube, GTFS shapes)
│   ├── tdk_demand_data.py   population grid, access-choice calibration, OD weights
│   └── tdk_tex_numbers.py   results → LaTeX macros
├── tdk/                 Hungarian TDK paper (tex, pdf, figures, generated numbers)
├── AUDIT.md             mathematical audit of v2
├── bp_markov_revised.tex / .pdf  Research paper
├── prezi_revised.tex / .pdf      Technical presentation
├── pyproject.toml
└── README.md
```

---

## Limitations

- **No AFC/APC validation**: demand estimates are prior-only and should be treated as illustrative.
- **Time-homogeneous CTMC**: valid within a single service window; use piecewise rates for diurnal variation.
- **No capacity constraints**: the scheduled departure hazard is independent of crowding and residual vehicle capacity.
- **Reducible resilience graphs**: Kemeny's constant is reported only for irreducible chains; fragmentation is scored explicitly after node removal.
- **Calendar semantics**: use `date_filter="YYYYMMDD"` for exact service. A weekday alone represents the union of services observed on that weekday.
- **Exact SSA**: infeasible at full BKK scale (~3.6 × 10⁹ events / 15 min); use τ-leaping.
- **Single-mode assignment**: passengers are assigned a mode at t=0; no mid-journey transfers modelled.

## Data and licensing

BKK's GTFS feed is downloaded at runtime and is not committed to this repository. The
source feed is published under CC0-1.0; the framework code is released under the MIT
License. Repository maintainer: Marcell Molnár.
