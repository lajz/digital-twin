```python
import numpy as np
from scipy.optimize import minimize

class Twin:
    FAMILY = "ltv-cac-steady-state"
    PARAMS = {
        "market":           (5e3, 5e5, "accounts"),
        "acq_per_dollar":   (1e-3, 5e-2, "accounts/$"),
        "monthly_churn":    (0.004, 0.15, "1/mo"),
        "arpa0":            (20.0, 400.0, "$/account/mo"),
        "arpa_growth":      (-0.01, 0.03, "1/mo"),
        "price_break":      (0.90, 1.30, "mult"),
        "rev_per_head":     (8e4, 4e5, "$/yr"),
        "opex_per_head":    (6e3, 2e4, "$/mo"),
        "fixed_opex":       (0.0, 2e5, "$/mo"),
        "headcount_smooth": (0.1, 0.9, "weight"),
        "market_growth":    (-0.01, 0.02, "1/mo"),
        "churn_scale":      (0.5, 2.0, "mult"),
        "acq_sat":          (0.5, 2.0, "mult"),
        "arpa_break_ramp":  (0.0, 0.05, "1/mo"),
        "lifetime_cap":     (5.0, 60.0, "months"),  # max retained age for LTV calc
    }
    METADATA = {
        "assumptions": [
            "LTV/CAC steady-state: new customers per month = spend / CAC, where CAC = acquisition cost per account",
            "CAC is derived from acq_per_dollar (1/acq_per_dollar), with market saturation dampening",
            "Churn is constant per month (exponential decay), no cohort aging effect",
            "ARPA drifts linearly, with a price break at month 30 (ramped)",
            "Headcount lags revenue via smoothed target",
            "Opex = headcount*opex_per_head + fixed_opex",
            "Cash is running total of net_income + capital_raised",
            "Market grows slowly over time",
            "Acquisition saturates as market fills (power-law dampening)",
            "Lifetime value = ARPA / churn, but we just model steady-state cohorts",
        ],
        "state_vars": ["customers", "cash"],
        "refs": [],
    }

    def fit(self, obs):
        # obs.time_s is in hours (720h = 1 month)
        t_months = obs.time_s / 720.0
        n = len(t_months)
        
        self.initial_customers = obs.customers[0]
        self.initial_cash = obs.cash[0]
        self.time_fit = t_months
        
        def objective(log_params):
            params = {}
            for i, k in enumerate(self.PARAMS):
                lo, hi, _ = self.PARAMS[k]
                params[k] = np.exp(log_params[i])
                params[k] = min(max(params[k], lo), hi)
            
            exog = {
                "marketing_spend": obs.marketing_spend,
                "capital_raised": obs.capital_raised,
            }
            time_s_full = obs.time_s * 3600.0  # hours -> seconds
            pred = self.predict(params, time_s_full, exog)
            
            err = 0.0
            err += 2.0 * np.mean((np.log(pred["mrr"] + 1) - np.log(obs.mrr + 1))**2)
            err += 2.0 * np.mean((np.log(pred["customers"] + 1) - np.log(obs.customers + 1))**2)
            err += np.mean(((pred["new_customers"] - obs.new_customers) / (obs.new_customers + 1))**2)
            err += np.mean(((pred["headcount"] - obs.headcount) / (obs.headcount + 1))**2)
            obs_ni = obs.net_income
            pred_ni = pred["net_income"]
            scale = np.abs(obs_ni) + 1e4
            err += np.mean(((pred_ni - obs_ni) / scale)**2)
            err += 2.0 * np.mean((np.log(pred["cash"] + 1) - np.log(obs.cash + 1))**2)
            return err

        # Initial guesses in log space (15 params)
        init = np.log([3e4, 0.01, 0.04, 90.0, 0.005, 1.2, 
                       2e5, 1e4, 5e4, 0.3, 0.005, 1.0, 1.0, 0.01, 20.0])
        bounds = [(np.log(lo), np.log(hi)) for lo, hi, _ in self.PARAMS.values()]
        
        best_params = None
        best_err = np.inf
        rng = np.random.default_rng(789)
        for _ in range(12):
            if _ > 0:
                init = np.log(np.array([
                    rng.uniform(2e4, 1e5),
                    rng.uniform(0.005, 0.03),
                    rng.uniform(0.02, 0.08),
                    rng.uniform(50, 150),
                    rng.uniform(0.0, 0.01),
                    rng.uniform(1.0, 1.3),
                    rng.uniform(1.5e5, 3e5),
                    rng.uniform(8e3, 1.5e4),
                    rng.uniform(2e4, 1e5),
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.0, 0.01),
                    rng.uniform(0.8, 1.2),
                    rng.uniform(0.8, 1.5),
                    rng.uniform(0.0, 0.03),
                    rng.uniform(10, 30),
                ]))
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-12})
            if res.fun < best_err:
                best_err = res.fun
                best_params = {k: np.exp(res.x[i]) for i, k in enumerate(self.PARAMS)}
        
        for k, (lo, hi, _) in self.PARAMS.items():
            best_params[k] = min(max(best_params[k], lo), hi)
        return best_params

    def predict(self, params, time_s, exog):
        # Convert seconds to months (1 month = 2592000 s)
        t_months = time_s / 2592000.0
        n = len(t_months)
        
        market0 = params["market"]
        acq_per_dollar = params["acq_per_dollar"]
        monthly_churn = params["monthly_churn"]
        arpa0 = params["arpa0"]
        arpa_growth = params["arpa_growth"]
        price_break = params["price_break"]
        rev_per_head = params["rev_per_head"]
        opex_per_head = params["opex_per_head"]
        fixed_opex = params["fixed_opex"]
        headcount_smooth = params["headcount_smooth"]
        market_growth = params["market_growth"]
        churn_scale = params["churn_scale"]
        acq_sat = params["acq_sat"]
        arpa_break_ramp = params["arpa_break_ramp"]
        lifetime_cap = params["lifetime_cap"]
        
        spend = np.asarray(exog["marketing_spend"], dtype=float)
        capital = np.asarray(exog["capital_raised"], dtype=float)
        
        customers = np.zeros(n)
        new_customers = np.zeros(n)
        mrr = np.zeros(n)
        headcount = np.zeros(n)
        net_income = np.zeros(n)
        cash = np.zeros(n)
        
        initial_customers = getattr(self, 'initial_customers', 0.0)
        initial_cash = getattr(self, 'initial_cash', 0.0)
        
        # Effective churn (scaled)
        churn = min(max(monthly_churn * churn_scale, 0.0), 1.0)
        retention = 1.0 - churn
        
        for t in range(n):
            # Market size grows slowly
            market = market0 * (1 + market_growth) ** t
            
            # Previous total customers
            if t == 0:
                prev_total = initial_customers
            else:
                prev_total = customers[t-1]
            
            # Acquisition: CAC-based, with saturation dampening
            # CAC = 1/acq_per_dollar (dollars per account)
            cac = 1.0 / acq_per_dollar
            sat_factor = max(1 - prev_total / market, 0.0)
            sat_factor = sat_factor ** acq_sat
            # New customers = spend / CAC, but capped by market saturation
            new_customers[t] = spend[t] / cac * sat_factor
            new_customers[t] = max(new_customers[t], 0.0)
            
            # Steady-state cohort model: customers = prev * retention + new
            if t == 0:
                customers[t] = initial_customers + new_customers[t]
            else:
                customers[t] = customers[t-1] * retention + new_customers[t]
            customers[t] = max(customers[t], 0.0)
            
            # ARPA with linear growth and price break (ramped in)
            arpa = arpa0 * (1 + arpa_growth * t)
            if t >= 30:
                ramp_frac = min(1.0, (t - 30) * arpa_break_ramp)
                arpa *= 1.0 + (price_break - 1.0) * ramp_frac
            mrr[t] = customers[t] * arpa
            
            # Headcount: smoothed target
            target_hc = mrr[t] * 12.0 / rev_per_head
            if t == 0:
                headcount[t] = max(target_hc, 1.0)
            else:
                headcount[t] = headcount_smooth * target_hc + (1 - headcount_smooth) * headcount[t-1]
            # Bound headcount changes
            if t > 0:
                headcount[t] = min(headcount[t], headcount[t-1] * 1.4)
                headcount[t] = max(headcount[t], headcount[t-1] * 0.6)
            headcount[t] = max(headcount[t], 0.0)
            
            # Net income and cash
            net_income[t] = mrr[t] - headcount[t] * opex_per_head - fixed_opex - spend[t]
            if t == 0:
                cash[0] = initial_cash + net_income[0] + capital[0]
            else:
                cash[t] = cash[t-1] + net_income[t] + capital[t]
        
        # Enforce non-negativity
        mrr = np.maximum(mrr, 0)
        customers = np.maximum(customers, 0)
        headcount = np.maximum(headcount, 0)
        new_customers = np.maximum(new_customers, 0)
        
        # Ensure customers <= previous + new (churn non-negative) – already satisfied
        # but enforce just in case
        for t in range(1, n):
            customers[t] = min(customers[t], customers[t-1] + new_customers[t])
        
        return {
            "mrr": mrr,
            "customers": customers,
            "new_customers": new_customers,
            "headcount": headcount,
            "net_income": net_income,
            "cash": cash,
        }
```