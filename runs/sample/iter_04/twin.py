import numpy as np
from scipy.spatial import cKDTree

class Twin:
    FAMILY = "overdamped-rods"
    PARAMS = {
        "elong_rate": (0.2, 3.0, "1/h"),
        "div_length": (3.0, 12.0, "um"),
        "div_cv":     (0.02, 0.35, "fraction"),
        "stiffness":  (0.5, 40.0, "um/(um*step)"),
    }
    METADATA = {
        "assumptions": [
            "Capsule-shaped cells with hemispherical caps; length is centerline length",
            "Exponential elongation with no crowding feedback (observed growth is close to exponential)",
            "Symmetric division at noisy target length; daughters inherit parent angle",
            "Soft repulsion between overlapping capsules, resolved by position-based dynamics",
            "Torque aligns contacting rods (nematic-like alignment)",
            "No death, no detachment; cells stay in 2D plane",
            "Width fixed at 1.0 um (E. coli width)",
            "Overdamped dynamics; inertial effects negligible",
            "Neighbor search via cKDTree with cutoff = 2*max_cell_length",
            "Internal timestep ~ 15 s, with multiple substeps per frame",
            "Cap cell count at 3000; early termination if too many cells",
        ],
        "state_vars": ["x","y","angle","length","width"],
        "refs": [
            "Rudge et al. 2012 (CellModeller)",
            "Ahmadi et al. 2024 (IPB benchmark)",
            "You et al. 2018 (E. coli colony growth)",
        ],
    }

    def fit(self, obs):
        times = obs.time_s / 3600.0  # hours
        counts = np.array([len(f["x"]) for f in obs.frames])
        total_len = np.array([np.sum(f["length"]) for f in obs.frames])
        
        # Growth rate from population count (exponential fit on log2)
        log2N = np.log2(counts)
        mask = times > 0.2 * times[-1]
        if np.sum(mask) > 2:
            slope, _ = np.polyfit(times[mask], log2N[mask], 1)
            doubling_time = 1.0 / slope
        else:
            doubling_time = 0.5
        elong_rate = np.log(2.0) / doubling_time
        
        # Division length: use 85th percentile of length distribution over all frames
        all_lengths = np.concatenate([f["length"] for f in obs.frames])
        div_length = np.percentile(all_lengths, 85)
        
        # CV of division length: from lengths near the division length
        high_len = all_lengths[all_lengths > np.percentile(all_lengths, 70)]
        if len(high_len) > 5:
            div_cv = np.std(high_len) / np.mean(high_len)
        else:
            div_cv = 0.1
        div_cv = np.clip(div_cv, 0.02, 0.35)
        
        # Stiffness: from mean NN distance and colony radius
        # Estimate from initial density: higher density -> stiffer repulsion
        init_nn = self._mean_nn(obs.frames[0])
        # Rough calibration: stiffness ~ 10 + 5*(2.5 - init_nn)
        stiffness = 10.0 + 5.0 * (2.5 - init_nn)
        stiffness = np.clip(stiffness, 0.5, 40.0)
        
        return {
            "elong_rate": float(elong_rate),
            "div_length": float(div_length),
            "div_cv": float(div_cv),
            "stiffness": float(stiffness),
        }

    def _mean_nn(self, frame):
        pts = np.stack([frame["x"], frame["y"]], axis=1)
        if len(pts) < 2:
            return 0.0
        tree = cKDTree(pts)
        dist, _ = tree.query(pts, k=2)
        return np.mean(dist[:, 1])

    def simulate(self, params, init_cells, time_s, seed):
        rng = np.random.default_rng(seed)
        
        elong_rate = params["elong_rate"] / 3600.0  # per second
        div_length = params["div_length"]
        div_cv = params["div_cv"]
        stiffness = params["stiffness"]  # um/(um*step) – effective spring constant
        
        # Internal timestep: 15 s (substeps per frame ~ 6 for 90 s frames)
        dt = 15.0  # seconds
        width = 1.0  # um, fixed
        
        # Initialize state
        x = init_cells["x"].copy().astype(np.float64)
        y = init_cells["y"].copy().astype(np.float64)
        angle = init_cells["angle"].copy().astype(np.float64)
        length = init_cells["length"].copy().astype(np.float64)
        w = np.full(len(x), width)
        
        # Precompute division targets for each cell
        # Each cell divides when length >= target (with noise)
        div_target = np.full(len(x), div_length)
        # Add noise per cell
        div_target *= (1.0 + rng.normal(0.0, div_cv, size=len(x)))
        
        # Output list
        frames_out = []
        frames_out.append({
            "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
            "length": length.copy(), "width": w.copy()
        })
        
        # Keep track of time
        current_time = time_s[0]
        frame_idx = 1
        
        # Main loop over frames
        while frame_idx < len(time_s):
            next_time = time_s[frame_idx]
            dt_frame = next_time - current_time
            n_steps = max(1, int(round(dt_frame / dt)))
            actual_dt = dt_frame / n_steps
            
            for _ in range(n_steps):
                # 1. Elongation (exponential)
                length *= np.exp(elong_rate * actual_dt)
                
                # 2. Division
                mask_div = length >= div_target
                if np.any(mask_div):
                    # Create daughter cells
                    n_div = np.sum(mask_div)
                    new_x = np.empty(n_div)
                    new_y = np.empty(n_div)
                    new_angle = np.empty(n_div)
                    new_length = np.empty(n_div)
                    new_w = np.full(n_div, width)
                    
                    idx_div = np.where(mask_div)[0]
                    for i, idx in enumerate(idx_div):
                        # Split length in half (keep total length conserved)
                        L = length[idx]
                        half = L / 2.0
                        # Parent becomes one daughter, new cell is the other
                        # Place daughter at same position + small offset along axis
                        dx = np.cos(angle[idx]) * half * 0.5
                        dy = np.sin(angle[idx]) * half * 0.5
                        # Parent stays at same position, new cell offset
                        new_x[i] = x[idx] + dx
                        new_y[i] = y[idx] + dy
                        new_angle[i] = angle[idx]
                        new_length[i] = half
                        # Update parent
                        length[idx] = half
                        # Reassign division target for parent (new target)
                        div_target[idx] = div_length * (1.0 + rng.normal(0.0, div_cv))
                    
                    # Append new cells
                    x = np.concatenate([x, new_x])
                    y = np.concatenate([y, new_y])
                    angle = np.concatenate([angle, new_angle])
                    length = np.concatenate([length, new_length])
                    w = np.concatenate([w, new_w])
                    div_target = np.concatenate([div_target, 
                                                 div_length * (1.0 + rng.normal(0.0, div_cv, size=n_div))])
                
                # 3. Repulsion and alignment (position-based dynamics)
                if len(x) > 1:
                    # Build KDTree with cutoff = 2*max_length + margin
                    max_L = np.max(length)
                    cutoff = 2.0 * max_L + 1.0
                    tree = cKDTree(np.stack([x, y], axis=1))
                    pairs = tree.query_pairs(cutoff, output_type='ndarray')
                    
                    if len(pairs) > 0:
                        # Compute overlaps and forces
                        n_pairs = len(pairs)
                        # Preallocate force arrays
                        fx = np.zeros(len(x))
                        fy = np.zeros(len(x))
                        torque = np.zeros(len(x))
                        
                        # For each pair, compute capsule overlap
                        for i, j in pairs:
                            # Vector between centers
                            dx = x[j] - x[i]
                            dy = y[j] - y[i]
                            dist = np.hypot(dx, dy)
                            if dist < 1e-6:
                                continue
                            
                            # Capsule half-lengths
                            hi = length[i] / 2.0
                            hj = length[j] / 2.0
                            
                            # Effective radius (capsule radius = width/2 = 0.5)
                            r_eff = 1.0  # sum of radii = 1.0 (width=1 each)
                            
                            # Approximate capsule distance: use center distance minus half-lengths projected
                            # Simple approximation: treat as spherocylinders
                            # Compute projection of vector onto each cell's axis
                            cos_i = np.cos(angle[i])
                            sin_i = np.sin(angle[i])
                            cos_j = np.cos(angle[j])
                            sin_j = np.sin(angle[j])
                            
                            # Project vector onto axes
                            proj_i = dx * cos_i + dy * sin_i
                            proj_j = dx * cos_j + dy * sin_j
                            
                            # Clamp to capsule extent
                            proj_i_clamped = np.clip(proj_i, -hi, hi)
                            proj_j_clamped = np.clip(proj_j, -hj, hj)
                            
                            # Closest points on each centerline
                            cx_i = x[i] + cos_i * proj_i_clamped
                            cy_i = y[i] + sin_i * proj_i_clamped
                            cx_j = x[j] + cos_j * proj_j_clamped
                            cy_j = y[j] + sin_j * proj_j_clamped
                            
                            dist_closest = np.hypot(cx_j - cx_i, cy_j - cy_i)
                            if dist_closest < 1e-6:
                                continue
                            
                            overlap = r_eff - dist_closest
                            if overlap > 0:
                                # Force magnitude proportional to overlap
                                force_mag = stiffness * overlap
                                # Direction: from i to j (normalized)
                                nx = (cx_j - cx_i) / dist_closest
                                ny = (cy_j - cy_i) / dist_closest
                                
                                # Apply equal and opposite forces
                                fx[i] += force_mag * nx
                                fy[i] += force_mag * ny
                                fx[j] -= force_mag * nx
                                fy[j] -= force_mag * ny
                                
                                # Torque: align cells along the line connecting them
                                # For nematic alignment, torque proportional to sin(2*(angle_i - angle_line))
                                angle_line = np.arctan2(ny, nx)
                                # Torque on i: align to line (but nematic: angle mod pi)
                                diff_i = angle_line - angle[i]
                                # Wrap to [-pi/2, pi/2]
                                diff_i = (diff_i + np.pi/2) % np.pi - np.pi/2
                                torque[i] += force_mag * 0.1 * np.sin(2*diff_i)
                                diff_j = angle_line - angle[j]
                                diff_j = (diff_j + np.pi/2) % np.pi - np.pi/2
                                torque[j] -= force_mag * 0.1 * np.sin(2*diff_j)
                        
                        # Apply forces (position update)
                        # Overdamped: dx = force * dt / viscosity (viscosity=1)
                        x += fx * actual_dt
                        y += fy * actual_dt
                        # Apply torque
                        angle += torque * actual_dt * 0.1  # small torque coefficient
                
                # 4. Cap cell count
                if len(x) > 3000:
                    # Truncate to first 3000 (should not happen in fit window)
                    x = x[:3000]
                    y = y[:3000]
                    angle = angle[:3000]
                    length = length[:3000]
                    w = w[:3000]
                    div_target = div_target[:3000]
            
            # Record frame
            frames_out.append({
                "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
                "length": length.copy(), "width": w.copy()
            })
            
            current_time = next_time
            frame_idx += 1
        
        return frames_out