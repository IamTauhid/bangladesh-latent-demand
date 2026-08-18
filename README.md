# Interval-Censored Learning for Latent Demand and Dispatch Composition

Interval-censored demand recovery and compositional dispatch forecasting for rationed
power systems. Research code for a study of the Bangladesh national grid (PGCB hourly records,
April 2015 – March 2026) targeting *IEEE Transactions on Sustainable Energy /
Smart Grid*.

## What this does

A coupled three-layer framework for grids whose published demand series is
determined by supply:

| Layer | Question | Method |
|---|---|---|
| **L1** | What would demand have been without rationing? | Partial identification: model-free bounds + an independent structural counterfactual |
| **L2** | How much comes from each fuel? | Merit-order ILR basis + Dirichlet deep regression on the simplex |
| **L3** | What does it cost, and who pays? | Fuel-wise cost → T&D loss → revenue → subsidy gap |

## Central empirical findings

1. **The published demand series is an accounting identity.** In 90.1% of pre-2022
   hours, `demand_mw` is *identically* `generation_mw`. It is a right-censored
   observation of latent demand, not a measurement of it.
2. **Rationing was flat across the day, not peak-clipping.** Shedding probability
   varied only 37–53% across the 24 hours in 2022–24, and the load-duration curve
   shows no truncation → the binding constraint was fuel affordability, not
   capacity adequacy. This invalidates the standard peak-truncation correction.
3. **A point estimate of suppressed demand is not identified.** A Tobit likelihood
   applies a near-constant 2.2–3.3 pp correction regardless of the truth. An
   interval-censored likelihood is far better calibrated (2.25 → 0.65 pp) and tracks
   curtailment *depth* almost perfectly (r ≥ 0.94), but is structurally blind to how
   much was *concealed* — at a 20% admission rate it recovers under half the truth.
   We therefore report bounds: 2023 suppression ∈ **[3.0, 9.9] pp** of served energy,
   with an independent structural estimate of **5.6 pp** after real-tariff and
   activity controls. The officially admitted 2.97 pp sits at the interval floor.
4. **The obvious confound runs the other way.** Nominal tariffs rose 22% over the
   crisis, but CPI inflation outpaced them: the **real tariff fell 13.7%**, so price
   response biases the suppression estimate *downward*, not upward.
5. **The energy–cost wedge**: liquid fuel supplied 22.9% of energy but 51.1% of
   cost in 2021. The bottom-up cost reconstruction matches the utility's published
   FY2025 average purchase cost to within 4.6%.

## Pipeline

Run in order from the repository root:

```bash
python src/01_clean.py         # physics-informed cleaning (energy-balance identity)
python src/03_weather.py       # NASA POWER, 8 divisions, population-weighted
python src/04_features.py      # calendar + Islamic calendar + weather + regime covariates
python src/05_censored.py      # Exp 1  : Tobit synthetic-censoring recovery
python src/05b_lambda.py       # Exp 1b : penalty / anchor-size sensitivity (diagnosis)
python src/05c_interval.py     # Exp 1c : Tobit vs INTERVAL censoring, known ground truth
python src/06_forecast.py      # L1 demand forecasting benchmark (Tasks A and B)
python src/07_composition.py   # L2 compositional fuel-mix forecasting
python src/08_economics.py     # L3 economics + counterfactual + sensitivity
python src/09_projection.py    # latent demand + projection to 2030 (interval-censored)
python src/10_figures.py       # publication figures -> paper/figs/
python src/11_tables.py        # LaTeX tables    -> paper/tables.tex
python src/12_identity_table.py# accounting-identity table (appends to tables.tex)
python src/13_crossref.py      # resolve bibliography entries via the Crossref API
python src/14_split_floats.py  # split tables.tex into one file per float
python src/15_place_floats.py  # position each float next to its first citation
python src/16_alpha_sensitivity.py  # sensitivity to alpha_min
python src/17_identification.py     # partial identification + confound controls
python src/18_identification_table.py  # -> paper/tabs/ident.tex
python src/19_sweep_magnitude.py   # extended synthetic sweep (0-4.9 pp)
```

Stages 10 and 11 write figures and LaTeX tables into `paper/`, which is not
tracked here (see below).


`02_eda.py` is exploratory and not required by the pipeline.

## Data

- **Primary**: `pgcb_V1_raw.csv` — Shekh & Rafi, *Hourly Electricity Generation,
  Demand, Load Shedding, and Fuel Mix Dataset for Bangladesh (2015–2026)*,
  Mendeley Data V1, doi:[10.17632/vpk8spw2mm.1](https://doi.org/10.17632/vpk8spw2mm.1),
  CC BY 4.0. 94,934 hourly records.
- **Weather**: NASA POWER daily API, 8 Bangladesh divisions, population-weighted
  by BBS 2022 census. Fetched by `src/03_weather.py` (no API key required).
- **Economic parameters**: published BPDB/BERC figures, listed in
  `src/08_economics.py` and swept in sensitivity analysis.

### Cleaning audit

The dataset is high quality. The energy-balance identity Σ(fuels + imports) =
generation holds with a **median absolute residual of 5 MW** (94.8% of hours within
50 MW). Rejected: 246 hours breaching physical ceilings, 478 hours whose energy
balance is violated by >10% of generation. Result: **3,927 clean days**, of which
27.5% are censored by load shedding. Full audit in `results/clean_audit.json`.

## Known limitations (stated in the paper)

- **Distribution loss is not in this dataset.** PGCB is the transmission operator.
  Distribution loss (7.25% FY24) and transmission loss (3.13% FY24) enter Layer 3
  as *exogenous published parameters*, interpolated between reported fiscal years
  and swept in sensitivity. No distribution-loss modelling is claimed.
- **Latent demand is counterfactual.** No measurement of unsuppressed demand
  exists. Validation is by controlled synthetic censoring plus identification from
  the uncensored 2016–2021 anchor era.
- **The 2030 projection was cut from the manuscript.** `09_projection.py` still
  produces it (and the latent-demand series the paper needs), but forward
  projection is scenario analysis on assumed weather and commissioning, not a
  validated result, so it is not reported.
- **Task B uses realised weather** over the evaluation window. There is no target
  leakage, but it measures the structural weather/calendar response under perfect
  meteorological foresight, not an operational forecast. Stated in the paper.

## Environment

Python 3.12 with pandas, numpy, scikit-learn, xgboost, torch (CPU), statsmodels,
scipy, matplotlib. No GPU required; the full pipeline runs on CPU in well under an
hour. `torch.set_num_threads(4)` is set in every model script — removing it causes
severe thread thrashing on small tensors.

## Layout

```
pgcb_V1_raw.csv          raw input
src/                     pipeline (numbered, run in order)
data/                    cleaned daily series, weather, feature matrix
results/                 metrics, audits, predictions (json/csv/npz)
```

## Manuscript

The manuscript is **not** included in this repository while it is under
preparation. Every figure and table in it is generated from the files in
`results/` by `src/10_figures.py` and `src/11_tables.py`, so the quantitative
content is fully reproducible from what is here.

`data/hourly_clean.csv` (8.7 MB) is also omitted; `src/01_clean.py` regenerates
it from `pgcb_V1_raw.csv` in a few seconds.
