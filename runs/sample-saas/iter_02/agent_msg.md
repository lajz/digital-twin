```python
import numpy as np
from scipy.optimize import minimize

class Twin:
    FAMILY = "bass-diffusion-churn"
    PARAMS = {
        "market":           (5e3, 2e5, "accounts"),
        "acq_per_dollar":   (2e-3, 4e-2, "accounts/$"),
        "monthly_churn":    (4e-3, 0.15, "1/mo"),
        "arpa0":            (20.0, 400.0, "$/account/mo"),
        "arpa_growth":      (-0.01, 0.03, "1/mo"),
        "rev_per_head":     (8e4, 4e5, "$/yr"),
        "opex_per_head":    (6e3, 2e4, "$/mo"),
        "fixed_opex":       (0.0, 2e5, "$/mo"),
        "price_break":      (0.90, 1.30, "mult"),  # multiplicative price change at month 30
        "headcount_smooth": (0.1, 0.9, "weight"),  # smoothing factor for headcount
    }
    METADATA = {
        "assumptions": [
            "Saturating acquisition: new = eff * spend * (1 - customers/market)",
            "Constant monthly churn rate",
            "ARPA drifts exponentially (arpa_growth per month)",
            "Price break at month 30 (x1.2) captured by price_break multiplier",
            "Headcount lags revenue: target = mrr*12/rev_per_head, smoothed",
            "Opex = headcount*opex_per_head + fixed_opex",
            "Cash is running total of net_income + capital_raised",
        ],
        "state_vars": ["customers", "cash"],
        "refs": [],
    }

    def fit(self, obs):
        # Convert hours to months (obs.time_s is in hours, 720h = 1 month)
        t_months = obs.time_s / 720.0
        n = len(t_months)
        
        # Use first observation as initial state (customers, cash)
        self.initial_customers = obs.customers[0]
        self.initial_cash = obs.cash[0]
        
        # Fit params using least squares on all observables
        def objective(log_params):
            params = {}
            for i, k in enumerate(self.PARAMS):
                lo, hi, _ = self.PARAMS[k]
                params[k] = np.exp(log_params[i])
                params[k] = min(max(params[k], lo), hi)
            
            # Build exog from obs (full horizon, but obs only has fit window)
            exog = {
                "marketing_spend": obs.marketing_spend,
                "capital_raised": obs.capital_raised,
            }
            # Use full time_s in seconds (obs.time_s is hours -> convert)
            time_s_full = obs.time_s * 3600.0
            pred = self.predict(params, time_s_full, exog)
            
            # Weighted residuals (log scale for mrr, customers, cash; relative for others)
            err = 0.0
            # MRR: log error
            err += 2.0 * np.mean((np.log(pred["mrr"] + 1) - np.log(obs.mrr + 1))**2)
            # Customers: log error
            err += 2.0 * np.mean((np.log(pred["customers"] + 1) - np.log(obs.customers + 1))**2)
            # New customers: relative error
            err += np.mean(((pred["new_customers"] - obs.new_customers) / (obs.new_customers + 1))**2)
            # Headcount: relative error
            err += np.mean(((pred["headcount"] - obs.headcount) / (obs.headcount + 1))**2)
            # Net income: relative error (careful with signs)
            obs_ni = obs.net_income
            pred_ni = pred["net_income"]
            scale = np.abs(obs_ni) + 1e4
            err += np.mean(((pred_ni - obs_ni) / scale)**2)
            # Cash: log error (cash is positive)
            err += 2.0 * np.mean((np.log(pred["cash"] + 1) - np.log(obs.cash + 1))**2)
            return err

        # Initial guesses in log space
        init = np.log([1e4, 1e-2, 0.03, 100.0, 0.005, 2e5, 1e4, 5e4, 1.2, 0.3])
        bounds = [(np.log(lo), np.log(hi)) for lo, hi, _ in self.PARAMS.values()]
        
        # Multiple restarts to avoid local minima
        best_params = None
        best_err = np.inf
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _ > 0:
                init = np.log(np.array([
                    rng.uniform(5e3, 2e5),
                    rng.uniform(2e-3, 4e-2),
                    rng.uniform(4e-3, 0.15),
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
        
        # Clamp final params
        for k, (lo, hi, _) in self.PARAMS.items():
            best_params[k] = min(max(best_params[k], lo), hi)
        return best_params

    def predict(self, params, time_s, exog):
        # Convert seconds to months (1 month = 2592000 s)
        t_months = time_s / 2592000.0
        n = len(t_months)
        
        # Unpack params
        market = params["market"]
        acq_per_dollar = params["acq_per_dollar"]
        monthly_churn = params["monthly_churn"]
        arpa0 = params["arpa0"]
        arpa_growth = params["arpa_growth"]
        rev_per_head = params["rev_per_head"]
        opex_per_head = params["opex_per_head"]
        fixed_opex = params["fixed_opex"]
        price_break = params["price_break"]
        headcount_smooth = params["headcount_smooth"]
        
        # Marketing spend and capital raised
        spend = np.asarray(exog["marketing_spend"], dtype=float)
        capital = np.asarray(exog["capital_raised"], dtype=float)
        
        # Initialize arrays
        customers = np.zeros(n)
        new_customers = np.zeros(n)
        mrr = np.zeros(n)
        headcount = np.zeros(n)
        net_income = np.zeros(n)
        cash = np.zeros(n)
        
        # Initial state (from fit, but default to 0 if not set)
        initial_customers = getattr(self, 'initial_customers', 0.0)
        initial_cash = getattr(self, 'initial_cash', 0.0)
        
        # Simulate month by month
        for t in range(n):
            # Acquisition
            if t == 0:
                # Use initial customers as starting point
                prev_customers = initial_customers
                # First month's new customers based on spend and saturation
                new_customers[t] = acq_per_dollar * spend[t] * (1 - prev_customers / market)
                customers[t] = prev_customers + new_customers[t] - monthly_churn * prev_customers
            else:
                prev_customers = customers[t-1]
                new_customers[t] = acq_per_dollar * spend[t] * (1 - prev_customers / market)
                customers[t] = prev_customers + new_customers[t] - monthly_churn * prev_customers
            # Ensure non-negative
            customers[t] = max(customers[t], 0.0)
            new_customers[t] = max(new_customers[t], 0.0)
            
            # ARPA: drift plus price break at month 30 (index 30 if t_months[0]==0)
            arpa = arpa0 * (1 + arpa_growth) ** t
            if t >= 30:
                arpa *= price_break
            mrr[t] = customers[t] * arpa
            
            # Headcount: target = mrr*12/rev_per_head, smoothed
            target_hc = mrr[t] * 12.0 / rev_per_head
            if t == 0:
                headcount[t] = target_hc
            else:
                headcount[t] = headcount_smooth * target_hc + (1 - headcount_smooth) * headcount[t-1]
            # Enforce max 40% monthly jump
            if t > 0:
                headcount[t] = min(headcount[t], headcount[t-1] * 1.4)
            headcount[t] = max(headcount[t], 0.0)
            
            # Net income
            net_income[t] = mrr[t] - headcount[t] * opex_per_head - fixed_opex - spend[t]
            
            # Cash
            if t == 0:
                cash[0] = initial_cash + net_income[0] + capital[0]
            else:
                cash[t] = cash[t-1] + net_income[t] + capital[t]
        
        # Enforce non-negativity for key variables
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