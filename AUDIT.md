# Mathematical audit of the BKK Markov passenger-flow framework (v2.2 → v3.0)

Scope: the code in `bkk/`, the paper `bp_markov_revised.tex` (v2.2) and the
presentation, audited against the pinned BKK feed **2572.20260604**
(valid 2026-06-02 … 2026-07-04, downloaded from the MobilityData mirror of
`go.bkk.hu`). Service day used throughout: **Tuesday 2026-06-09**, window 07:00–09:00.

Severity: **A** = invalidates results, **B** = biases results materially,
**C** = presentation / rigour. "Fixed" points to the change in this revision.

The corrected model is in `bkk/linemodel.py`; the Hungarian TDK paper in
`tdk/tdk_dolgozat.tex` gives the full derivations.

---

## 1. Model specification

| # | Sev | Finding | Consequence | Resolution |
|---|-----|---------|-------------|------------|
| M1 | A | **Every passenger alights at every stop.** State = stop, transition = one inter-stop hop, after which the passenger waits again for a new vehicle. | A 10-stop ride incurs 10 independent waits. Stationary occupancy is mostly people "waiting" mid-journey. | Augmented state space X = H ∪ S: hub waiting states plus in-vehicle segment states. Continuing on the same line is a segment→segment transition with GTFS continuation shares r_{ss'}. |
| M2 | A | **In-vehicle time is absent from the dynamics.** The holding time at stop i is Exp(μ_i) (waiting only); τ_ij enters only the softmax cost. | Travel is instantaneous, so the KFE time scale is the headway alone. | Segment states have holding rate 1/τ_s. |
| M3 | B | **Waiting hazard = vehicle frequency f.** For random passenger arrivals the mean wait is E[W] = E[H](1+CV²)/2 (renewal/inspection paradox). With exponential waiting at rate f the mean is 1/f = E[H], twice the value for regular service. | Waits are overstated ≈2× (median headway CV² on the feed is 0.02). | Moment-matched hazard θ = 2f/(1+CV²), with CV² estimated per segment from GTFS departure times. |
| M4 | B | **The entropy choice acts only where a stop has ≥ 2 successors, and it chooses the next stop, not a vehicle.** Measured on 2026-06-09 with the v2 builder: 19.4 % of stops branch (30.2 % of departures): bus 20.5 %, tram 6.1 %, trolleybus 9.3 %, metro 85 %, HÉV 69 %. At those stops it does split flow (mean 1 − max p ≈ 0.49). But (a) at 80 % of stops π = 1 regardless of λ; (b) at metro/HÉV stations the "branches" are mostly the two travel directions (after parent-station aggregation), chosen by the cost of the next hop instead of the destination; (c) there is no choice between modes, because each mode is a separate chain. So Wilson's *trip-distribution* interpretation does not apply. | Choice is made at **hubs** (which merge all platforms and modes): in v3, 82 % of active hubs offer ≥ 2 segments, covering 95.7 % of boardings, and 40.6 % of boardings happen where ≥ 2 modes are available. Re-derived as **minimum relative entropy** (Snickars–Weibull) w.r.t. the frequency-share prior: π_{s\|h} ∝ θ_s e^{−λ c_s}, realised exactly by Poisson thinning q_s = θ_s a_s. λ = 0 recovers the common-lines frequency share (Chriqui–Robillard, Spiess–Florian). |
| M5 | B | **Rate q_ij = f_i · π_{j\|i}.** A passenger waiting for direction j still leaves at the aggregate frequency of all directions. | The sojourn ignores which vehicles are acceptable, which is inconsistent with M4. | q_s = θ_s a_s: the total boarding rate Σ_s q_s ≤ Σ_s θ_s, so waits grow when vehicles are rejected. |
| M6 | C | **The transfer term is row-constant:** δ·t_tf(i) is added to every out-edge of i, so it cancels in the softmax. κ^v is a mode constant inside a single-mode chain, so only the product λκ^v is identified. | The parameters are not identifiable, and the transfer penalty has no effect. | The joint multimodal hub network makes κ^v comparable across modes at a hub. The cost is defined per km (c_s = κ^v·1000·τ_s/d_s) so that long segments are not penalised for covering more distance. |
| M7 | A | **Closed population with no sinks.** Passengers circulate forever; "Σ N_i(0) = peak boardings" equates a **stock** with a **flow**. | Absolute counts are meaningless; Little's law (L = λW) is violated. | Open network: Poisson origin streams λ_h, exit after alighting with probability 1−p_tr. Stocks follow from flows by L = λ(−T)^{-1}. |
| M8 | C | "Single-mode assignment"; modes are independent chains. | Interchanges (Deák, Széll K.) cannot matter. | Hubs merge platforms (parent_station) and same-name stops within 400 m across modes. |

## 2. Probability / stochastic-process claims

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| P1 | B | "By linearity **and the mean-field approximation** the KFE holds for counts"; "KFE recovered as N→∞". For a population of independent passengers (first-order channels) E[N(t)] = N(0)e^{Qt} is **exact for every N**. | Stated as an exact identity. Furthermore, for Poisson inflow the occupancies are **independent Poisson** at every t (an M_t/G/∞-type linear network). With a fixed initial population the occupancies are multinomial. So the full distribution is closed-form, and SSA / τ-leap ensembles are needed only as verification. |
| P2 | B | The Poisson-departure assumption is "justified when vehicles run at roughly constant headway". This is backwards: constant headway gives Uniform(0, H) waits. | See M3. |
| P3 | A | **Importance I_i = π_i μ_i mixes two chains.** π comes from the embedded jump chain P. The CTMC stationary law is π^Q_i ∝ π^P_i/μ_i, so the departure flux is π^Q_i μ_i ∝ π^P_i; the old formula weights by μ_i twice. | `ResilienceAnalyser.departure_flux` (fixed, tested). |
| P4 | B | **Kemeny truncation:** 20 leading eigenvalues plus "neutral zero tail" (+1 per missing eigenvalue). For n in the thousands the tail is not neutral, and K can be biased by orders of magnitude. | Dense exact spectrum for n ≤ 1500. In model v3 the Kemeny constant is replaced by quantities with time units (mean journey time, demand-weighted efficiency). |
| P5 | B | **Spectral gap of P** is 0 for periodic chains: two-way lines are bipartite, giving the eigenvalue −1. | Any CTMC measure avoids periodicity. On the real network the slowest mode of T is **localised** at a stop served once per window (relaxation time ≈ 2.9 h), so the gap measures the worst-served corner, not the system. We report time-to-95 % of the transient instead. |
| P6 | C | Ergodicity is assumed per mode, but SCC partitioning is described and not implemented. Dead ends were made absorbing, so the chain was reducible and K = NaN. | In the open model (−T) is a nonsingular M-matrix (every state reaches the exit). No irreducibility assumption is needed. |
| P7 | C | The entropy proposition uses μ_i for a Lagrange multiplier (clashes with the hazard), and claims a single λ from one aggregate cost constraint while writing λ^v. | Re-derived cleanly in the paper (§3). |

## 3. Numerical methods

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| N1 | A | **τ-leap step τ = ε·a₀/max a_r is dimensionless** (a ratio of propensities), and in practice it is always clipped to 10·τ₀, so "τ = 1 s" ran with 10 s leaps. | Cao–Gillespie–Petzold bound for first-order channels: τ = min(τ₀, ε / max_{N_i>0} μ_i) (fixed, tested). |
| N2 | B | **Efficiency sampled the first 200 stop indices** (sorted stop-id order, not random), with unweighted hops. | Exact all-sources BFS in the legacy module. Model v3 uses exact Dijkstra on expected times (wait 1/θ + ride τ), weighted by OD demand. |
| N3 | C | Betweenness with k = 500 sampled sources and no seed is irreproducible. | `seed=0`. |
| N4 | C | Criticality = geometric mean of min–max normalised scores: the lowest candidate is 0 by construction, and the scale depends on the candidate set. | Replaced by the relative efficiency loss ΔE/E, split into total and network (other OD pairs) effects. |

## 4. Data handling

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| D1 | A | **Route types 11 (trolleybus) and 109 (HÉV) unknown.** The current feed no longer uses 800/2, so both modes were silently dropped. | Added to `ROUTE_TYPES` (fixed, tested). |
| D2 | A | **Weekday filter = union over all matching dates.** On the 2026 feed: 96 033 trips for "tuesday" vs 39 662 on 2026-06-09, i.e. frequencies ×2.42 (M2: 1654 vs 590). | The analysis uses `date_filter`; a warning is logged for weekday-only use. |
| D3 | B | Peak share φ = 0.11 described as a one-hour share but applied to the two-hour 07–09 window; the module's own diurnal profile sums to 1.28 instead of 1. | φ₀₇₋₀₉ = 0.20; profile normalised. Under the linear model all flows scale with N_day·φ, so only the scale is assumed. |
| D4 | C | The paper says platform aggregation bypasses surface modes; the code applies it globally. | Documented; v3 aggregates deliberately. |

## 5. Paper integrity

* **Withdrawn numbers still printed** (Tables 7, 8, 9, 11; the 117 319 vs 440 000 discrepancy). The revised paper reports only reproducible numbers from `scripts/tdk_analysis.py`.
* **Citations that do not support the claim or could not be verified** and are removed: `schiewe2022` (the source of the β prior means; not found), `yildirimoglu2018` (author list in text ≠ bibliography; title not verifiable), `bliemer2003` (venue and year inconsistent), `zhang2019` (cited for subway dwell times, but the title is about traveller information), `jaiswal1961` (a priority queue, cited for station queues), `heidergott2010` (max-plus algebra, cited as a CTMC queueing precedent), `helbing2000` (escape panic, cited for Markov pedestrian flow), `maerivoet2005` (cellular automata, cited for Gillespie).
* Reproducibility checklist referenced non-existent scripts (`gtfs_ingest.py`, …).

## 6. Verification added

`tests/test_linemodel.py` (18 tests): sub-generator structure, the headway moment match, the λ = 0 frequency-share limit, flow balance and Little's law, linear scaling, transient convergence, **Monte Carlo journeys vs the exact solution**, closure vs failure semantics, and regressions for P3, N1, N2 and D1. The full suite: 101 tests pass.
