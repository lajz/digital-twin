```python
import numpy as np
from scipy.optimize import minimize

class Twin:
    FAMILY = "cohort-retention"
    PARAMS = {
        "market":           (5e3, 2e5, "accounts"),
        "acq_per_dollar":   (2e-3, 4e-2, "accounts/$"),
        "retention_alpha":  (0.5, 0.99, "1/mo"),      # retention decay shape (power-law)
        "retention_beta":   (0.0, 0.2, "1/mo"),       # retention decay rate (exponential component)
        "arpa0":            (20.0, 400.0, "$/account/mo"),
        "arpa_growth":      (-0.01, 0.03, "1/mo"),
        "rev_per_head":     (8e4, 4e5, "$/yr"),
        "opex_per_head":    (6e3, 2e4, "$/mo"),
        "fixed_opex":       (0.0, 2e5, "$/mo"),
        "price_break":      (0.90, 1.30, "mult"),     # multiplicative price change at month 30
        "headcount_smooth": (0.1, 0.9, "weight"),
    }
    METADATA = {
        "assumptions": [
            "Cohort-based retention: each monthly cohort decays with a power-law+exponential curve",
            "Saturating acquisition: new = eff * spend * (1 - customers/market)",
            "ARPA drifts exponentially, with a price break at month 30",
            "Headcount lags revenue: target = mrr*12/rev_per_head, smoothed",
            "Opex = headcount*opex_per_head + fixed_opex",
            "Cash is running total of net_income + capital_raised",
        ],
        "state_vars": ["customers", "cash", "cohorts"],
        "refs": [],
    }

    def fit(self, obs):
        # Convert hours to months (obs.time_s is in hours, 720h = 1 month)
        t_months = obs.time_s / 720.0
        n = len(t_months)
        
        # Use first observation as initial state
        self.initial_customers = obs.customers[0]
        self.initial_cash = obs.cash[0]
        
        # Fit params using least squares on all observables
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
            time_s_full = obs.time_s * 3600.0
            pred = self.predict(params, time_s_full, exog)
            
            # Weighted residuals (log scale for mrr, customers, cash; relative for others)
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

        # Initial guesses in log space
        init = np.log([1e4, 1e-2, 0.9, 0.05, 100.0, 0.005, 2e5, 1e4, 5e4, 1.2, 0.3])
        bounds = [(np.log(lo), np.log(hi)) for lo, hi, _ in self.PARAMS.values()]
        
        best_params = None
        best_err = np.inf
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _ > 0:
                init = np.log(np.array([
                    rng.uniform(5e3, 2e5),
                    rng.uniform(2e-3, 4e-2),
                    rng.uniform(0.5, 0.99),
                    rng.uniform(0.0, 0.2),
                    rng.uniform(20, 400),
                    rng.uniform(-0.01, 0.03),
                    rng.uniform(8e4, 4e5),
                    rng.uniform(6e3, 2e4),
                    rng.uniform(0, 2e5),
                    rng.uniform(0.9, 1.3),
                    rng.uniform(0.1, 0.9),
                ]))
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 300, "ftol": 1e-10})
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
        
        market = params["market"]
        acq_per_dollar = params["acq_per_dollar"]
        retention_alpha = params["retention_alpha"]
        retention_beta = params["retention_beta"]
        arpa0 = params["arpa0"]
        arpa_growth = params["arpa_growth"]
        rev_per_head = params["rev_per_head"]
        opex_per_head = params["opex_per_head"]
        fixed_opex = params["fixed_opex"]
        price_break = params["price_break"]
        headcount_smooth = params["headcount_smooth"]
        
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
        
        # Cohort matrix: cohorts[cohort_month, age] = surviving customers
        # We'll track up to n cohorts
        cohorts = np.zeros((n, n))
        
        # Initialize: first month's cohort from initial customers (assume they arrived in month 0)
        # But we'll treat initial customers as a pre-existing cohort
        if n > 0:
            # Initial cohort: all initial customers at age 0
            cohorts[0, 0] = initial_customers
        
        for t in range(n):
            # Acquisition for this month
            # Saturation based on current total customers (sum of all cohorts)
            if t == 0:
                prev_total = initial_customers
            else:
                prev_total = customers[t-1]
            
            # New customers this month
            new_customers[t] = acq_per_dollar * spend[t] * (1 - prev_total / market)
            new_customers[t] = max(new_customers[t], 0.0)
            
            # Add new cohort (age 0)
            cohorts[t, 0] = new_customers[t]
            
            # Age existing cohorts: apply retention decay
            # For each cohort j < t, its survivors at age (t-j) = initial * retention(t-j)
            # We compute retention as power-law * exponential
            for j in range(t):
                age = t - j
                if age > 0:
                    # Retention from previous age to current age
                    # retention(age) = alpha^age * exp(-beta*age)
                    # But we need incremental decay: cohort[j, age] = cohort[j, age-1] * decay(age)
                    # decay(age) = retention(age) / retention(age-1)
                    if age == 1:
                        decay = retention_alpha * np.exp(-retention_beta)
                    else:
                        # retention(age) = alpha^age * exp(-beta*age)
                        # retention(age-1) = alpha^(age-1) * exp(-beta*(age-1))
                        # ratio = alpha * exp(-beta)
                        decay = retention_alpha * np.exp(-retention_beta)
                    cohorts[j, age] = cohorts[j, age-1] * decay
            
            # Total customers = sum of all cohorts at their current age
            total = 0.0
            for j in range(t+1):
                age = t - j
                total += cohorts[j, age]
            customers[t] = total
            customers[t] = max(customers[t], 0.0)
            
            # ARPA with drift and price break
            arpa = arpa0 * (1 + arpa_growth) ** t
            if t >= 30:
                arpa *= price_break
            mrr[t] = customers[t] * arpa
            
            # Headcount
            target_hc = mrr[t] * 12.0 / rev_per_head
            if t == 0:
                headcount[t] = target_hc
            else:
                headcount[t] = headcount_smooth * target_hc + (1 - headcount_smooth) * headcount[t-1]
            if t > 0:
                headcount[t] = min(headcount[t], headcount[t-1] * 1.4)
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
        
        return {
            "mrr": mrr,
            "customers": customers,
            "new_customers": new_customers,
            "headcount": headcount,
            "net_income": net_income,
            "cash": cash,
        }
```