"""A synthetic seed-stage SaaS business as a `Domain`.

Proves the loop is domain-portable: same feedback loop, a different subject. The twin
models customer acquisition / churn / expansion / headcount / cash; it is handed the
business's *planned* levers (marketing spend, capital raised) over the whole horizon and
must produce a forecast that both fits the history and obeys the cash-flow identity
exactly -- "free plausibility" from accounting, no tuning.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from medusa import prompts
from medusa.contract.interface import Observations, Task
from medusa.data import build
from medusa.domains.base import Domain, bounded_step, flow_balance, non_negative, ratio_within

MONTH_S = 30.0 * 86400.0

SAAS_TASK = Task(
    name="saas",
    mode="series",
    observables=("mrr", "customers", "net_income", "cash", "headcount", "new_customers"),
    weights={"mrr": 1.5, "cash": 1.2, "customers": 1.0, "net_income": 0.8,
             "headcount": 0.6, "new_customers": 0.6},
    exogenous=("marketing_spend", "capital_raised"),
    plausibility={"monthly_churn": [0.004, 0.15], "arpa_usd": [20.0, 600.0]},
    period_s=MONTH_S,
    period_label="mo",
)


@dataclasses.dataclass(frozen=True, slots=True)
class SaaSPreset:
    name: str
    months: int
    market: float          # serviceable customers
    k_acq: float           # customers acquired per $ of spend at zero saturation
    churn: float           # monthly logo churn
    arpa0: float           # $ / customer / month
    arpa_growth: float     # monthly
    price_step_month: int
    price_step_mult: float
    rev_per_head: float    # annual revenue per employee the org staffs toward
    opex_per_head: float   # $ / employee / month
    fixed_opex: float      # $ / month
    hc_alpha: float        # headcount smoothing
    cash0: float
    customers0: float
    seed: int


PRESETS = {
    "saas-seed": SaaSPreset(
        name="saas-seed", months=48, market=40000.0, k_acq=0.011, churn=0.025,
        arpa0=90.0, arpa_growth=0.003, price_step_month=30, price_step_mult=1.2,
        rev_per_head=180000.0, opex_per_head=11000.0, fixed_opex=25000.0,
        hc_alpha=0.25, cash0=800000.0, customers0=400.0, seed=1,
    ),
    # high churn + a small market: growth stalls near saturation, no price step
    "saas-plateau": SaaSPreset(
        name="saas-plateau", months=42, market=9000.0, k_acq=0.014, churn=0.055,
        arpa0=140.0, arpa_growth=0.001, price_step_month=999, price_step_mult=1.0,
        rev_per_head=160000.0, opex_per_head=12000.0, fixed_opex=40000.0,
        hc_alpha=0.3, cash0=1_200_000.0, customers0=600.0, seed=2,
    ),
}


def _levers(p: SaaSPreset) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(p.months)
    # spend: ramp 40k -> 180k over 24 months, budget cut to 120k at month 36
    spend = 40000.0 + (180000.0 - 40000.0) * np.clip(t / 24.0, 0, 1)
    spend[t >= 36] = 120000.0
    rng = np.random.default_rng(p.seed + 7)
    spend *= rng.lognormal(0.0, 0.03, p.months)
    capital = np.zeros(p.months)
    capital[12] = 6_000_000.0   # Series A
    return spend, capital


def generate(p: SaaSPreset) -> Observations:
    rng = np.random.default_rng(p.seed)
    spend, capital = _levers(p)
    n = p.months

    cust = np.zeros(n); newc = np.zeros(n); arpa = np.zeros(n)
    mrr = np.zeros(n); hc = np.zeros(n); ni = np.zeros(n); cash = np.zeros(n)

    cust[0] = p.customers0
    hc[0] = max(6.0, p.customers0 * p.arpa0 * 12 / p.rev_per_head)
    for i in range(n):
        step = p.price_step_mult if i >= p.price_step_month else 1.0
        arpa[i] = p.arpa0 * (1 + p.arpa_growth) ** i * step
        if i > 0:
            newc[i] = p.k_acq * spend[i] * max(0.0, 1.0 - cust[i - 1] / p.market)
            cust[i] = cust[i - 1] + newc[i] - p.churn * cust[i - 1]
            target_hc = cust[i] * arpa[i] * 12 / p.rev_per_head
            hc[i] = hc[i - 1] + p.hc_alpha * (target_hc - hc[i - 1])
        else:
            newc[i] = p.k_acq * spend[i]
        mrr[i] = cust[i] * arpa[i]
        ni[i] = mrr[i] - hc[i] * p.opex_per_head - p.fixed_opex - spend[i]
        cash[i] = (cash[i - 1] if i else p.cash0) + ni[i] + capital[i]

    # observation noise on the measured series; net_income / cash stay consistent with
    # the *noisy* mrr so the accounting identity still holds in the data.
    def noisy(x, cv):
        return x * rng.lognormal(0.0, cv, n)

    mrr_o = noisy(mrr, 0.04)
    cust_o = np.round(noisy(cust, 0.03))
    newc_o = np.clip(np.round(noisy(newc, 0.08)), 0, None)
    hc_o = np.round(noisy(hc, 0.02))
    ni_o = mrr_o - hc_o * p.opex_per_head - p.fixed_opex - spend
    cash_o = p.cash0 + np.cumsum(ni_o + capital)

    t_s = np.arange(n) * MONTH_S
    return Observations(
        time_s=t_s, population_count=cust_o,   # population_count kept = customers (contract)
        extra={
            "mrr": mrr_o, "customers": cust_o, "new_customers": newc_o,
            "headcount": hc_o, "net_income": ni_o, "cash": cash_o,
            "marketing_spend": spend, "capital_raised": capital,
        },
    )


def datasheet(p: SaaSPreset) -> str:
    return f"""\
# Datasheet: {p.name}  (business / SaaS)

- **Source:** synthetic (medusa.domains.saas). A seed-stage B2B SaaS, {p.months} monthly
  periods. Mechanistic model + observation noise; the accounting identities hold exactly.
- **Scored channels:** mrr, customers, net_income, cash, headcount, new_customers ($ / month
  or counts).
- **Given levers (exogenous, provided over the FULL horizon incl. holdout):**
  `marketing_spend` -- ramps $40k -> $180k over 2 years, then a budget cut to $120k at
  month 36. `capital_raised` -- a $6M Series A lands in month 12.
- **Structural break:** a list-price increase (x{p.price_step_mult:g}) at month {p.price_step_month}.
- **Enforced identities (hard constraints -- a twin that breaks one is rejected):**
  - `cash[t] = cash[t-1] + net_income[t] + capital_raised[t]`
  - `customers[t] <= customers[t-1] + new_customers[t]` (churn is non-negative)
  - mrr, customers, headcount, new_customers are non-negative
  - implied ARPA `mrr / customers` stays in a sane band; headcount changes are bounded
- **Plausible monthly logo churn:** {p.churn:g} (band [0.004, 0.15]).
"""


def build_saas(preset_name: str = "saas-seed", *, fit_frac: float = 0.6, **kw) -> build.Dataset:
    p = PRESETS[preset_name]
    obs = generate(p)
    ground_truth = {
        "monthly_churn": p.churn,
        "churn_plausible": [0.004, 0.15],
        "arpa_usd_plausible": [20.0, 600.0],
        "market": p.market,
    }
    return build.write(
        obs, datasheet(p), name=preset_name, ground_truth=ground_truth,
        fit_frac=fit_frac, task=SAAS_TASK, **kw,
    )


# --- constraints ----------------------------------------------------------------


def _customer_flow(pred, _ds):
    from medusa.domains.base import ConstraintResult

    c, nc = pred.get("customers"), pred.get("new_customers")
    if c is None or nc is None:
        return ConstraintResult(True, "customer_flow", "channel absent")
    over = (c[1:] - c[:-1]) - nc[1:]
    bad = np.where(over > 0.02 * np.clip(c[:-1], 1.0, None) + 5.0)[0]
    if bad.size:
        k = int(bad[0]) + 1
        return ConstraintResult(
            False, "customer_flow",
            f"at t={k}: net adds {c[k]-c[k-1]:.0f} exceed gross adds "
            f"new_customers={nc[k]:.0f} (churn cannot be negative)",
        )
    return ConstraintResult(True, "customer_flow")


_customer_flow.__name__ = "customer_flow"

SAAS_CONSTRAINTS = (
    non_negative("mrr", "customers", "headcount", "new_customers"),
    flow_balance("cash", "net_income", extra_inflows=("capital_raised",),
                 rel_tol=0.02, abs_tol=25_000.0),
    _customer_flow,
    ratio_within("mrr", "customers", 15.0, 800.0),
    bounded_step("headcount", 0.55),  # a smoothed hiring model can still step at month 0-1
)


# --- register -----------------------------------------------------------------

# in-run critic default: off (`kind="business"` -> `critic.resolve_enabled` says no). A
# 2026-09-19 5-dataset A/B (`medusa bench --iters 8 --temperature 0.35`) found the coach
# stance regressed saas-seed's combined holdout sMAPE (0.068 off -> 0.115 coach), well
# past the ~0.016 honest-bar noise band measured on the *aggregate* multi-dataset score.
# An n=1 same-day saas-seed-only re-check read that as critic-caused family-hopping, but
# noted the critic-off baseline in that same session was independently just as unstable
# -- so it couldn't tell whether the critic caused the instability or just rode on top of
# a domain that's unstable either way.
#
# A follow-up multi-replicate saas-seed-only A/B settled it (2026-09-19, `--iters 8
# --temperature 0.35`, 5 reps/condition, response cache disabled per-rep so each is a
# genuinely fresh trajectory, not a cache replay): the 5-dataset regression does NOT
# replicate at the single-dataset level.
#
#   condition   mean best_holdout_smape (sd)   mean distinct families   mean valid_rate
#   off         0.131  (sd 0.063)              3.2                      0.90
#   coach       0.106  (sd 0.025)               3.0                     0.925
#   skeptic     0.139  (sd 0.059)               3.2                     0.95
#
# Every pairwise mean gap (off-coach 0.025, off-skeptic 0.009, coach-skeptic 0.034) is
# well under 1x the larger of the two conditions' own within-condition stdev -- run-to-
# run noise on this one dataset dwarfs any critic effect. The "critic causes family-
# hopping instability" hypothesis doesn't hold up either: mean distinct families explored
# is essentially flat across conditions (coach explored *fewer* on average, not more).
# saas-seed is just a high-variance target regardless of the critic (the off condition
# alone ranged 0.077-0.228 holdout sMAPE across its 5 reps).
#
# Verdict: critic on/off is a wash on SaaS, not a proven regression -- but a wash isn't a
# proven win either, so the default stays off: no measured benefit to justify the extra
# LLM calls. Revisit only if some future prompt/config change gives the critic a clear,
# replicated edge on this domain.


def _register() -> None:
    from medusa.domains import register

    for name in PRESETS:
        register(Domain(
            name=name,
            task=SAAS_TASK,
            build=lambda *, _n=name, **kw: build_saas(_n, **kw),
            system_prompt=prompts.SAAS_SYSTEM_PROMPT,
            blurb="synthetic seed-stage SaaS business (MRR / churn / cash / headcount)",
            kind="business",
            constraints=SAAS_CONSTRAINTS,
        ))


_register()
