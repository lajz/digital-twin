# Portfolio -- ipb-ecoli-spatial  (task: spatial)

## 1. stochastic-rod-population  (iteration 6)

- combined holdout sMAPE: **0.1435**
- per-observable: total_length_um 0.225, population_count 0.116, colony_radius_um 0.009, colony_aspect 0.043, nematic_order 0.551, mean_nn_dist_um 0.070
- implied doubling time: 0.84 h (plausibility 1.00)
- parameters: `{"elong_rate": 0.8270147535567426, "div_length": 5.160129703719579, "div_cv": 0.1807990088484257, "repulsion_strength": 0.05612563476412165, "align_rate": 1.2340674339548299, "radial_drift": 0.5, "noise_scale": 2.0}`

## 2. overdamped-rods  (iteration 1)

- combined holdout sMAPE: **0.2015**
- per-observable: total_length_um 0.225, population_count 0.135, colony_radius_um 0.026, colony_aspect 0.050, nematic_order 0.104, mean_nn_dist_um 0.544
- implied doubling time: 0.84 h (plausibility 1.00)
- parameters: `{"elong_rate": 0.8270147535567426, "div_length": 5.498206894584007, "div_cv": 0.17255251303061772, "stiffness": 5.0}`

## 3. pore-channel-network  (iteration 12)

- combined holdout sMAPE: **0.2787**
- per-observable: total_length_um 0.344, population_count 0.184, colony_radius_um 0.056, colony_aspect 0.175, nematic_order 0.192, mean_nn_dist_um 0.626
- implied doubling time: 0.91 h (plausibility 1.00)
- parameters: `{"elong_rate": 0.7655113347280801, "div_length": 5.160129703719579, "div_cv": 0.1807990088484257, "channel_k": 0.12251269528243292}`
