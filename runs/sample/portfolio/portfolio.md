# Portfolio -- synthetic-ecoli-fast

## 1. baranyi-ode  (iteration 3)

- holdout sMAPE: **0.0579**
- holdout MASE: 2.310
- fit R^2: 0.9911
- implied doubling time: 0.35 h (plausibility 1.00)
- parameters: `{"n0": 1.12289507303528, "q0": 1.3594072747682648, "mu_max": 1.9820810731422331, "nmax": 39791.71367294471}`

## 2. richards  (iteration 4)

- holdout sMAPE: **0.0660**
- holdout MASE: 2.623
- fit R^2: 0.9903
- implied doubling time: 0.37 h (plausibility 1.00)
- parameters: `{"carrying_capacity": 38028.15223444709, "nu": 1.3147234662994243, "mu_max": 2.496165084253447, "t_infl": 5.541608330679521}`

## 3. logistic  (iteration 1)

- holdout sMAPE: **0.0930**
- holdout MASE: 3.781
- fit R^2: 0.9856
- implied doubling time: 0.37 h (plausibility 1.00)
- parameters: `{"n0": 1.0000000000000002, "mu_max": 1.859620589129036, "carrying_capacity": 42940.60697521882}`
