```python
import numpy as np
from scipy.optimize import minimize

class Twin:
    FAMILY = "cohort-retention"
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
        "retention_aging":  (0.5, 1.5, "mult"),  # multiplier on churn for older cohorts
        "acq_sat":          (0.5, 2.0, "mult"),
        "arpa_break_ramp":  (0.0, 0.05, "1/mo"),
    }
    METADATA = {
        "assumptions": [
            "Cohort retention: each monthly cohort decays at a rate that increases with age",
            "Acquisition is spend-driven with market saturation dampening",
            "ARPA drifts linearly, with a price break at month 30 (ramped)",
            "Headcount lags revenue via smoothed target",
            "Opex = headcount*opex_per_head + fixed_opex",
            "Cash is running total of net_income + capital_raised",
            "Churn increases for older cohorts (retention_aging > 1 means more churn for old)",
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

        # Initial guesses in log space (13 params)
        init = np.log([3e4, 0.01, 0.04, 90.0, 0.005, 1.2, 
                       2e5, 1e4, 5e4, 0.3, 1.0, 1.0, 0.01])
        bounds = [(np.log(lo), np.log(hi)) for lo, hi, _ in self.PARAMS.values()]
        
        best_params = None
        best_err = np.inf
        rng = np.random.default_rng(123)
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
                    rng.uniform(0.8, 1.2),
                    rng.uniform(0.8, 1.5),
                    rng.uniform(0.0, 0.03),
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
        retention_aging = params["retention_aging"]
        acq_sat = params["acq_sat"]
        arpa_break_ramp = params["arpa_break_ramp"]
        
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
        
        # Track cohorts: each is an array of surviving customers per month
        # We'll store cohorts as a list of arrays (one per month)
        cohorts = []  # each element is np.array of survivors from that cohort's birth month
        
        for t in range(n):
            # Market size (constant in this model)
            market = market0
            
            # Previous total customers
            if t == 0:
                prev_total = initial_customers
            else:
                prev_total = customers[t-1]
            
            # Acquisition: spend-driven with saturation
            cac = 1.0 / acq_per_dollar
            sat_factor = max(1 - prev_total / market, 0.0)
            sat_factor = sat_factor ** acq_sat
            new_customers[t] = spend[t] / cac * sat_factor
            new_customers[t] = max(new_customers[t], 0.0)
            
            # Add new cohort
            cohorts.append(np.array([new_customers[t]]))
            
            # Age existing cohorts: apply age-dependent churn
            total_customers = 0.0
            if t == 0:
                # Initial customers as a pre-existing cohort
                total_customers = initial_customers + new_customers[t]
                # We'll treat initial customers as a cohort that started at t=0
                # (simplification: just add them to total)
            else:
                # Age each existing cohort
                aged_cohorts = []
                for age, cohort in enumerate(cohorts[:-1]):  # exclude new one
                    # Churn increases with age
                    age_factor = 1.0 + (retention_aging - 1.0) * min(age, 24) / 24.0
                    churn_eff = monthly_churn * age_factor
                    retention_eff = 1.0 - min(churn_eff, 1.0)
                    survivors = cohort * retention_eff
                    aged_cohorts.append(survivors)
                    total_customers += np.sum(survivors)
                # Add new cohort
                total_customers += new_customers[t]
                # Update cohorts list (excluding the new one which we added earlier)
                cohorts = aged_cohorts + [np.array([new_customers[t]])]
            
            customers[t] = total_customers
            
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