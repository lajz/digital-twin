import numpy as np
from scipy.optimize import minimize

class Twin:
    FAMILY = "cohort-retention"
    PARAMS = {
        "market":           (5e3, 2e5, "accounts"),
        "acq_per_dollar":   (2e-3, 4e-2, "accounts/$"),
        "retention_alpha":  (0.5, 0.99, "1/mo"),
        "retention_beta":   (0.0, 0.2, "1/mo"),
        "arpa0":            (20.0, 400.0, "$/account/mo"),
        "arpa_growth":      (-0.01, 0.03, "1/mo"),
        "arpa_accel":       (-0.001, 0.001, "1/mo^2"),
        "price_break":      (0.90, 1.30, "mult"),
        "rev_per_head":     (8e4, 4e5, "$/yr"),
        "opex_per_head":    (6e3, 2e4, "$/mo"),
        "fixed_opex":       (0.0, 2e5, "$/mo"),
        "headcount_smooth": (0.1, 0.9, "weight"),
        "market_growth":    (-0.01, 0.02, "1/mo"),
        "churn_scale":      (0.5, 2.0, "mult"),       # multiplier on base churn
    }
    METADATA = {
        "assumptions": [
            "Cohort-based retention: each cohort decays with power-law+exponential curve",
            "Saturating acquisition: new = eff * spend * (1 - customers/market)",
            "ARPA drifts with quadratic time trend, with a price break at month 30",
            "Headcount lags revenue: target = mrr*12/rev_per_head, smoothed",
            "Opex = headcount*opex_per_head + fixed_opex",
            "Cash is running total of net_income + capital_raised",
            "Market grows slowly over time",
            "Churn rate scales with cohort age (younger cohorts churn faster)",
        ],
        "state_vars": ["customers", "cash", "cohorts"],
        "refs": [],
    }

    def fit(self, obs):
        # obs.time_s is in hours (720h = 1 month)
        t_months = obs.time_s / 720.0
        n = len(t_months)
        
        self.initial_customers = obs.customers[0]
        self.initial_cash = obs.cash[0]
        
        # Time vector in months for fitting
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
            
            # Weighted residuals
            err = 0.0
            # Log-scale for mrr, customers, cash (relative errors)
            err += 2.0 * np.mean((np.log(pred["mrr"] + 1) - np.log(obs.mrr + 1))**2)
            err += 2.0 * np.mean((np.log(pred["customers"] + 1) - np.log(obs.customers + 1))**2)
            err += np.mean(((pred["new_customers"] - obs.new_customers) / (obs.new_customers + 1))**2)
            err += np.mean(((pred["headcount"] - obs.headcount) / (obs.headcount + 1))**2)
            
            # Net income: relative error with scale
            obs_ni = obs.net_income
            pred_ni = pred["net_income"]
            scale = np.abs(obs_ni) + 1e4
            err += np.mean(((pred_ni - obs_ni) / scale)**2)
            
            err += 2.0 * np.mean((np.log(pred["cash"] + 1) - np.log(obs.cash + 1))**2)
            return err

        # Initial guesses in log space (based on observed data)
        # market ~ 20k, acq ~ 0.01, retention_alpha ~ 0.9, beta ~ 0.05
        # arpa0 ~ 90, arpa_growth ~ 0.005, arpa_accel ~ 0, price_break ~ 1.2
        # rev_per_head ~ 2e5, opex_per_head ~ 1e4, fixed_opex ~ 5e4
        # headcount_smooth ~ 0.3, market_growth ~ 0.005, churn_scale ~ 1.0
        init = np.log([2e4, 0.01, 0.9, 0.05, 90.0, 0.005, 0.0, 1.2, 
                       2e5, 1e4, 5e4, 0.3, 0.005, 1.0])
        bounds = [(np.log(lo), np.log(hi)) for lo, hi, _ in self.PARAMS.values()]
        
        best_params = None
        best_err = np.inf
        rng = np.random.default_rng(42)
        for _ in range(8):
            if _ > 0:
                init = np.log(np.array([
                    rng.uniform(1e4, 5e4),
                    rng.uniform(0.005, 0.02),
                    rng.uniform(0.8, 0.98),
                    rng.uniform(0.01, 0.1),
                    rng.uniform(50, 150),
                    rng.uniform(0.0, 0.01),
                    rng.uniform(-0.0005, 0.0005),
                    rng.uniform(1.0, 1.3),
                    rng.uniform(1.5e5, 3e5),
                    rng.uniform(8e3, 1.5e4),
                    rng.uniform(2e4, 1e5),
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.0, 0.01),
                    rng.uniform(0.8, 1.2),
                ]))
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 400, "ftol": 1e-12})
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
        retention_alpha = params["retention_alpha"]
        retention_beta = params["retention_beta"]
        arpa0 = params["arpa0"]
        arpa_growth = params["arpa_growth"]
        arpa_accel = params["arpa_accel"]
        price_break = params["price_break"]
        rev_per_head = params["rev_per_head"]
        opex_per_head = params["opex_per_head"]
        fixed_opex = params["fixed_opex"]
        headcount_smooth = params["headcount_smooth"]
        market_growth = params["market_growth"]
        churn_scale = params["churn_scale"]
        
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
        # We'll track age in months from cohort start
        cohorts = np.zeros((n, n))
        
        # Initialize: first month's cohort from initial customers (assume they arrived in month 0)
        if n > 0:
            cohorts[0, 0] = initial_customers
        
        # Precompute decay factors: retention(age) = alpha^age * exp(-beta*age)
        # decay from age-1 to age = retention(age)/retention(age-1) = alpha * exp(-beta)
        # But we want churn to be higher for younger cohorts, so we scale by age
        # Use a churn that decreases with age: churn(age) = base_churn * exp(-churn_scale * age)
        # decay(age) = 1 - churn(age)
        max_age = n
        decay_factors = np.zeros(max_age)
        for age in range(1, max_age):
            # Base churn from retention params: churn = 1 - alpha * exp(-beta)
            base_churn = 1.0 - retention_alpha * np.exp(-retention_beta)
            # Churn decreases with age
            churn_age = base_churn * np.exp(-churn_scale * age)
            decay_factors[age] = 1.0 - churn_age
        
        for t in range(n):
            # Market size grows slowly
            market = market0 * (1 + market_growth) ** t
            
            # Acquisition for this month
            if t == 0:
                prev_total = initial_customers
            else:
                prev_total = customers[t-1]
            
            # New customers this month (saturating)
            new_customers[t] = acq_per_dollar * spend[t] * max(1 - prev_total / market, 0.0)
            new_customers[t] = max(new_customers[t], 0.0)
            
            # Add new cohort (age 0)
            if t < n:
                cohorts[t, 0] = new_customers[t]
            
            # Age existing cohorts: apply decay factors
            # For each cohort j < t, age = t - j
            for j in range(t):
                age = t - j
                if age > 0 and age < max_age:
                    cohorts[j, age] = cohorts[j, age-1] * decay_factors[age]
                elif age >= max_age:
                    # Very old cohorts: assume they die off
                    cohorts[j, age] = 0.0
            
            # Total customers = sum of all cohorts at their current age
            total = 0.0
            for j in range(t+1):
                age = t - j
                if age < max_age:
                    total += cohorts[j, age]
            customers[t] = max(total, 0.0)
            
            # ARPA with quadratic drift and price break
            arpa = arpa0 * (1 + arpa_growth * t + 0.5 * arpa_accel * t**2)
            if t >= 30:
                arpa *= price_break
            mrr[t] = customers[t] * arpa
            
            # Headcount: smoothed target
            target_hc = mrr[t] * 12.0 / rev_per_head
            if t == 0:
                headcount[t] = target_hc
            else:
                headcount[t] = headcount_smooth * target_hc + (1 - headcount_smooth) * headcount[t-1]
            # Bound headcount changes
            if t > 0:
                headcount[t] = min(headcount[t], headcount[t-1] * 1.4)
                headcount[t] = max(headcount[t], headcount[t-1] * 0.6)
            headcount[t] = max(headcount[t], 0.0)
            
            # Net income and cash (accounting identity)
            net_income[t] = mrr[t] - headcount[t] * opex_per_head - fixed_opex - spend[t]
            if t == 0:
                cash[0] = initial_cash + net_income[0] + capital[0]
            else:
                cash[t] = cash[t-1] + net_income[t] + capital[t]
        
        # Enforce non-negativity and constraints
        mrr = np.maximum(mrr, 0)
        customers = np.maximum(customers, 0)
        headcount = np.maximum(headcount, 0)
        new_customers = np.maximum(new_customers, 0)
        net_income = np.maximum(net_income, -np.inf)  # can be negative
        
        # Ensure customers <= previous + new (churn non-negative)
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