# Factor Models and Implied Volatility in Portfolio Optimization

Code accompanying an MBA thesis on portfolio construction within Markowitz's mean-variance
framework. The central question: can injecting better economic information into the two
inputs that framework never directly observes — expected returns and risk — make the
resulting portfolios more *robust*, independently of whether they outperform on any single
realized market path?

Two levers are tested on a 19-stock US large-cap universe (2015-2026):

- **Expected returns**: a Fama-French 6-factor model (market, SMB, HML, RMW, CMA, Momentum)
  in place of raw historical averages.
- **Risk**: a hybrid covariance matrix blending historical covariance with implied
  volatility extracted from live option prices via Black-Scholes inversion,
  $\Sigma_\lambda = (1-\lambda)\Sigma_{hist} + \lambda\Sigma_{impl}$.

Both are evaluated out-of-sample with a rolling backtest, a Ledoit-Wolf statistical test on
the Sharpe-ratio gap, and a battery of implied-volatility stress scenarios — plus a few
extensions (ESG-constrained optimization, a real-money tracked portfolio).

This repository holds the code only; the thesis write-up itself is not included here.

## Repository structure

**Core pipeline**
| File | Purpose |
|---|---|
| `config.py` | Universe, tickers, sector mapping, study period, global settings/paths |
| `data_utils.py` | Price downloading, return computation, data alignment |
| `Fama_French.py` | FF6 factor data, regressions, factor-based expected returns |
| `black_scholes.py` | Black-Scholes pricing, vega, Newton-Raphson implied-vol inversion |
| `implied_volatility.py` | ATM implied-vol extraction from live option chains |
| `covariance.py` | Historical / implied / shrunk hybrid covariance construction |
| `optimization.py` | Mean-variance optimization core (min-variance, max-Sharpe, ESG constraint) |
| `portfolio_builders.py` | End-to-end helper building one optimized portfolio |
| `portfolio_metrics.py` | Sharpe, Sortino, HHI/ENP, risk contributions, drawdown, rolling Sharpe |
| `efficient_frontier.py` | Efficient frontier and tangency/min-variance summaries |
| `array_utils.py` | Small array-coercion helper used across the pipeline |

**Evaluation & extensions**
| File | Purpose |
|---|---|
| `backtest.py` | Rolling out-of-sample backtest (historical vs. FF6, with an ESG variant) |
| `statistical_tests.py` | Ledoit-Wolf stationary-bootstrap test on the Sharpe-ratio gap |
| `stress_test.py` | Implied-vol stress scenarios (uniform, sectoral, historical-crisis, hierarchy inversion) |
| `esg_scores.py` | LSEG ESG score loading/formatting |
| `esg_weekly_tracking.py` | Real-money ESG-constrained portfolio, weekly mark-to-market tracking |
| `tsla_exclusion.py` | Standalone analysis justifying Tesla's exclusion from the universe |
| `unconstrained_example.py` | Closed-form unconstrained tangency portfolio (illustrative) |
| `comparison.py` | Small notebook-display formatting helper |

**`visualization.py`** holds every plotting/animation function used to produce the thesis's
figures.

**`notebook_final.ipynb`** ties the whole pipeline together end to end and is the best
starting point for following the analysis in order.

**`data/`** holds the frozen snapshots (`*_snapshot.csv`, the ESG `.xlsx`, the lambda-grid
archive) consumed by `load_or_fetch_*`-style functions. **`figures/`** holds every static
plot and mp4 animation the notebook generates.

## Getting started

```bash
pip install -r requirements.txt
```

The animation functions in `visualization.py` (the mp4 exports) additionally require
[ffmpeg](https://ffmpeg.org/) to be available on your `PATH`.

Then open `notebook_final.ipynb`, or import any module directly, e.g.:

```python
from data_utils import load_or_fetch_prices
from config import TICKERS, START_DATE, END_DATE, PRICES_SNAPSHOT_PATH

prices = load_or_fetch_prices(TICKERS, START_DATE, END_DATE, PRICES_SNAPSHOT_PATH)
```

## Reproducibility and data

All results are built from **frozen snapshots** (the `*_snapshot.csv` / `.json` files in
this repo), not from live re-downloads. `load_or_fetch_*`-style functions read the snapshot
if it exists and only hit the network to create it the first time. This keeps every number
and figure in the thesis reproducible from a fixed point in time, independent of how
`yfinance` data or option chains look when you happen to run the code.

## A known limitation: no historical implied volatility

Implied volatility is only available in real time — free data sources expose live option
chains, not historical ones, and paid historical-IV providers (OptionMetrics, ORATS, ...)
were outside the scope of this project. This means the hybrid covariance's forward-looking
component cannot be backtested the same way the factor-based expected returns are: there's
no multi-year IV history to roll through. The pipeline works around this with implied-vol
*stress scenarios* (`stress_test.py`) rather than a historical backtest for that piece — a
deliberate, documented choice rather than an oversight. Logging implied vol daily going
forward (mirroring `esg_weekly_tracking.py`'s pattern) would be a natural extension, but
would take years to accumulate a usable history.

## A note on AI-assisted development

A meaningful part of this codebase, particularly the visualization and animation code, was
written with the help of Claude Code. I was already comfortable with Python but
had no prior experience with most of the libraries this project required.
