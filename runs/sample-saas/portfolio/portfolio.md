# Portfolio -- saas-seed  (task: saas)

## 1. cohort-retention  (iteration 7)

- combined holdout sMAPE: **0.0763**
- per-observable: mrr 0.034, customers 0.030, net_income 0.182, cash 0.046, headcount 0.143, new_customers 0.112
- implied doubling time: nan h (plausibility 1.00)
- parameters: `{"market": 19059.29786641582, "acq_per_dollar": 0.010931549257454619, "churn_base": 0.04279092169803184, "churn_decay": 0.0010204961330431252, "arpa0": 92.14785708193203, "arpa_growth": 0.0003369559010207839, "price_break": 1.2793749887517725, "rev_per_head": 146159.99835692483, "opex_per_head": 9836.679546922358, "fixed_opex": 55192.345658450016, "headcount_smooth": 0.16379455778860477, "market_growth": 0.0116728492742656, "churn_scale": 0.563867498715274, "acq_sat": 0.5}`

## 2. funnel-conversion  (iteration 6)

- combined holdout sMAPE: **0.1215**
- per-observable: mrr 0.114, customers 0.022, net_income 0.375, cash 0.117, headcount 0.052, new_customers 0.045
- implied doubling time: nan h (plausibility 1.00)
- parameters: `{"market": 39713.31098587798, "visit_rate": 0.5488175783685028, "trial_rate": 0.1637955694045037, "convert_rate": 0.12142741851926656, "churn_base": 0.030769994649992922, "churn_decay": 0.00028797913922204465, "arpa0": 92.21023003089707, "arpa_growth": 0.00028879304373581393, "price_break": 1.146996411297674, "rev_per_head": 146192.74703803367, "opex_per_head": 9837.387651950865, "fixed_opex": 55184.39947858989, "headcount_smooth": 0.1638791661786701, "market_growth": 4.365159719743487e-09, "churn_scale": 0.7786440558665771}`

## 3. ltv-cac-steady-state  (iteration 9)

- combined holdout sMAPE: **0.2412**
- per-observable: mrr 0.097, customers 0.028, net_income 0.985, cash 0.232, headcount 0.127, new_customers 0.097
- implied doubling time: nan h (plausibility 1.00)
- parameters: `{"market": 26833.661961621045, "acq_per_dollar": 0.010102991686764958, "monthly_churn": 0.04584244775891152, "arpa0": 79.18477054786882, "arpa_growth": 0.009892445493763344, "price_break": 1.0968702876357597, "rev_per_head": 127119.24488350163, "opex_per_head": 10607.86197005523, "fixed_opex": 32134.923250621072, "headcount_smooth": 0.11235685762673035, "market_growth": 2.2955949230366757e-08, "churn_scale": 0.5017409382569407, "acq_sat": 0.5, "arpa_break_ramp": 0.02146611011708004, "lifetime_cap": 11.120442274742793}`
