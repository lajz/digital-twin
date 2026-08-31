import numpy as np
from scipy.spatial import cKDTree

class Twin:
    FAMILY = "stochastic-rod-population"
    PARAMS = {
        "elong_rate": (0.2, 3.0, "1/h"),
        "div_length": (3.0, 12.0, "um"),
        "div_cv":     (0.02, 0.35, "fraction"),
        "repulsion_strength": (0.0, 0.5, "um/step"),
        "align_rate": (0.0, 5.0, "1/h"),
        "radial_drift": (-0.5, 0.5, "um/h"),
        "noise_scale": (0.0, 2.0, "um/sqrt(h)"),
    }
    METADATA = {
        "assumptions": [
            "Capsule cells with hemispherical caps; length is centerline length",
            "Exponential elongation with fixed rate (no crowding feedback)",
            "Symmetric division at noisy target length; daughters inherit parent angle",
            "Soft repulsion between overlapping capsules, resolved by position-based dynamics",
            "Nematic alignment torque proportional to contact force",
            "Radial drift term to control colony expansion (captures boundary effects)",
            "Langevin-type noise on position (diffusive spreading)",
            "No death, no detachment; cells stay in 2D plane",
            "Width fixed at 1.0 um (E. coli width)",
            "Neighbor search via cKDTree with cutoff = 2*max_cell_length + margin",
            "Internal timestep ~ 15 s, with variable substeps per frame",
            "Cap cell count at 3000; early termination if too many cells",
        ],
        "state_vars": ["x","y","angle","length","width"],
        "refs": [
            "Rudge et al. 2012 (CellModeller)",
            "Ahmadi et al. 2024 (IPB benchmark)",
            "You et al. 2018 (E. coli colony growth)",
            "Volpe et al. 2014 (active matter)",
        ],
    }

    def fit(self, obs):
        times = obs.time_s / 3600.0  # hours
        counts = np.array([len(f["x"]) for f in obs.frames])
        total_len = np.array([np.sum(f["length"]) for f in obs.frames])
        radii = np.array([self._gyration_radius(f) for f in obs.frames])
        nematic = np.array([self._nematic_order(f) for f in obs.frames])
        
        # Growth rate from population count (exponential fit on log2, ignoring early lag)
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
        
        # Repulsion strength: from initial density and colony radius stability
        init_nn = self._mean_nn(obs.frames[0])
        # Higher density -> stronger repulsion; also colony radius growth slows with density
        repulsion_strength = 0.1 + 0.1 * (2.5 - init_nn)
        repulsion_strength = np.clip(repulsion_strength, 0.0, 0.5)
        
        # Alignment rate: from nematic order trend (decreasing with density)
        if len(nematic) > 5:
            slope_nem = np.polyfit(times, nematic, 1)[0]
            align_rate = np.clip(0.5 - slope_nem * 10.0, 0.0, 5.0)
        else:
            align_rate = 0.5
        
        # Radial drift: from colony radius slope (drift outward if radius grows)
        if len(radii) > 5:
            slope_r = np.polyfit(times, radii, 1)[0]
            radial_drift = np.clip(slope_r * 2.0, -0.5, 0.5)
        else:
            radial_drift = 0.0
        
        # Noise scale: from colony radius fluctuations around trend
        if len(radii) > 5:
            resid = radii - np.polyval(np.polyfit(times, radii, 1), times)
            noise_scale = np.clip(np.std(resid) * 10.0, 0.0, 2.0)
        else:
            noise_scale = 0.5
        
        return {
            "elong_rate": float(elong_rate),
            "div_length": float(div_length),
            "div_cv": float(div_cv),
            "repulsion_strength": float(repulsion_strength),
            "align_rate": float(align_rate),
            "radial_drift": float(radial_drift),
            "noise_scale": float(noise_scale),
        }

    def _mean_nn(self, frame):
        pts = np.stack([frame["x"], frame["y"]], axis=1)
        if len(pts) < 2:
            return 0.0
        tree = cKDTree(pts)
        dist, _ = tree.query(pts, k=2)
        return np.mean(dist[:, 1])

    def _gyration_radius(self, frame):
        if len(frame["x"]) == 0:
            return 0.0
        cx = np.mean(frame["x"])
        cy = np.mean(frame["y"])
        return np.sqrt(np.mean((frame["x"]-cx)**2 + (frame["y"]-cy)**2))

    def _nematic_order(self, frame):
        if len(frame["angle"]) == 0:
            return 0.0
        angles = frame["angle"]
        order = np.mean(np.exp(2j * angles))
        return np.abs(order)

    def simulate(self, params, init_cells, time_s, seed):
        rng = np.random.default_rng(seed)
        
        elong_rate = params["elong_rate"] / 3600.0  # per second
        div_length = params["div_length"]
        div_cv = params["div_cv"]
        repulsion_strength = params["repulsion_strength"]  # um/step
        align_rate = params["align_rate"] / 3600.0  # per second
        radial_drift = params["radial_drift"] / 3600.0  # um/s
        noise_scale = params["noise_scale"] / 60.0  # um/sqrt(s) (convert from sqrt(h))
        
        width = 1.0  # um, fixed
        
        # Internal timestep: aim for ~5 neighbor searches per frame
        dt = 18.0  # seconds
        
        # Initialize state
        x = init_cells["x"].copy().astype(np.float64)
        y = init_cells["y"].copy().astype(np.float64)
        angle = init_cells["angle"].copy().astype(np.float64)
        length = init_cells["length"].copy().astype(np.float64)
        w = np.full(len(x), width)
        
        # Precompute division targets for each cell
        div_target = div_length * (1.0 + rng.normal(0.0, div_cv, size=len(x)))
        
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
                # 1. Elongation (constant rate, no crowding feedback)
                length *= np.exp(elong_rate * actual_dt)
                
                # 2. Division
                mask_div = length >= div_target
                if np.any(mask_div):
                    n_div = np.sum(mask_div)
                    new_x = np.empty(n_div)
                    new_y = np.empty(n_div)
                    new_angle = np.empty(n_div)
                    new_length = np.empty(n_div)
                    new_w = np.full(n_div, width)
                    
                    idx_div = np.where(mask_div)[0]
                    for i, idx in enumerate(idx_div):
                        L = length[idx]
                        half = L / 2.0
                        # Place daughter at same position + small offset along axis
                        dx = np.cos(angle[idx]) * half * 0.5
                        dy = np.sin(angle[idx]) * half * 0.5
                        new_x[i] = x[idx] + dx
                        new_y[i] = y[idx] + dy
                        new_angle[i] = angle[idx]
                        new_length[i] = half
                        # Update parent
                        length[idx] = half
                        # Reassign division target for parent
                        div_target[idx] = div_length * (1.0 + rng.normal(0.0, div_cv))
                    
                    # Append new cells
                    x = np.concatenate([x, new_x])
                    y = np.concatenate([y, new_y])
                    angle = np.concatenate([angle, new_angle])
                    length = np.concatenate([length, new_length])
                    w = np.concatenate([w, new_w])
                    div_target = np.concatenate([div_target, 
                                                 div_length * (1.0 + rng.normal(0.0, div_cv, size=n_div))])
                
                # 3. Repulsion, alignment, and noise (position-based dynamics)
                if len(x) > 1:
                    # Build KDTree with cutoff = 2*max_length + margin
                    max_L = np.max(length)
                    cutoff = 2.0 * max_L + 1.0
                    pts = np.stack([x, y], axis=1)
                    tree = cKDTree(pts)
                    pairs = tree.query_pairs(cutoff, output_type='ndarray')
                    
                    if len(pairs) > 0:
                        # Preallocate force and torque arrays
                        fx = np.zeros(len(x))
                        fy = np.zeros(len(x))
                        torque = np.zeros(len(x))
                        
                        # Vectorized pair processing for speed
                        i_idx = pairs[:, 0]
                        j_idx = pairs[:, 1]
                        dx = x[j_idx] - x[i_idx]
                        dy = y[j_idx] - y[i_idx]
                        dist = np.hypot(dx, dy)
                        
                        # Avoid division by zero
                        valid = dist > 1e-6
                        if np.any(valid):
                            i_v = i_idx[valid]
                            j_v = j_idx[valid]
                            dx_v = dx[valid]
                            dy_v = dy[valid]
                            dist_v = dist[valid]
                            
                            # Capsule half-lengths
                            hi = length[i_v] / 2.0
                            hj = length[j_v] / 2.0
                            
                            # Compute closest points on each centerline
                            cos_i = np.cos(angle[i_v])
                            sin_i = np.sin(angle[i_v])
                            cos_j = np.cos(angle[j_v])
                            sin_j = np.sin(angle[j_v])
                            
                            # Project vector onto axes
                            proj_i = dx_v * cos_i + dy_v * sin_i
                            proj_j = dx_v * cos_j + dy_v * sin_j
                            
                            # Clamp to capsule extent
                            proj_i_clamped = np.clip(proj_i, -hi, hi)
                            proj_j_clamped = np.clip(proj_j, -hj, hj)
                            
                            # Closest points
                            cx_i = x[i_v] + cos_i * proj_i_clamped
                            cy_i = y[i_v] + sin_i * proj_i_clamped
                            cx_j = x[j_v] + cos_j * proj_j_clamped
                            cy_j = y[j_v] + sin_j * proj_j_clamped
                            
                            dist_closest = np.hypot(cx_j - cx_i, cy_j - cy_i)
                            valid2 = dist_closest > 1e-6
                            
                            if np.any(valid2):
                                i_v2 = i_v[valid2]
                                j_v2 = j_v[valid2]
                                dist_c = dist_closest[valid2]
                                cx_i2 = cx_i[valid2]
                                cy_i2 = cy_i[valid2]
                                cx_j2 = cx_j[valid2]
                                cy_j2 = cy_j[valid2]
                                
                                # Overlap
                                overlap = 1.0 - dist_c  # sum of radii = 1.0
                                pos_overlap = overlap > 0
                                
                                if np.any(pos_overlap):
                                    i_v3 = i_v2[pos_overlap]
                                    j_v3 = j_v2[pos_overlap]
                                    overlap_pos = overlap[pos_overlap]
                                    cx_i3 = cx_i2[pos_overlap]
                                    cy_i3 = cy_i2[pos_overlap]
                                    cx_j3 = cx_j2[pos_overlap]
                                    cy_j3 = cy_j2[pos_overlap]
                                    dist_c3 = dist_c[pos_overlap]
                                    
                                    # Force magnitude (stiffness scaled by repulsion_strength)
                                    force_mag = repulsion_strength * overlap_pos
                                    
                                    # Direction from i to j
                                    nx = (cx_j3 - cx_i3) / dist_c3
                                    ny = (cy_j3 - cy_i3) / dist_c3
                                    
                                    # Apply forces (equal and opposite)
                                    np.add.at(fx, i_v3, force_mag * nx)
                                    np.add.at(fy, i_v3, force_mag * ny)
                                    np.add.at(fx, j_v3, -force_mag * nx)
                                    np.add.at(fy, j_v3, -force_mag * ny)
                                    
                                    # Torque for alignment (nematic)
                                    angle_line = np.arctan2(ny, nx)
                                    diff_i = angle_line - angle[i_v3]
                                    diff_i = (diff_i + np.pi/2) % np.pi - np.pi/2
                                    diff_j = angle_line - angle[j_v3]
                                    diff_j = (diff_j + np.pi/2) % np.pi - np.pi/2
                                    
                                    torque_i = force_mag * align_rate * np.sin(2*diff_i) * actual_dt
                                    torque_j = -force_mag * align_rate * np.sin(2*diff_j) * actual_dt
                                    np.add.at(torque, i_v3, torque_i)
                                    np.add.at(torque, j_v3, torque_j)
                    
                    # Apply forces (overdamped: dx = force * dt)
                    x += fx * actual_dt
                    y += fy * actual_dt
                    # Apply torque
                    angle += torque * actual_dt
                
                # 4. Radial drift (outward if positive)
                if radial_drift != 0.0:
                    cx = np.mean(x)
                    cy = np.mean(y)
                    rx = x - cx
                    ry = y - cy
                    r = np.hypot(rx, ry) + 1e-6
                    x += radial_drift * rx / r * actual_dt
                    y += radial_drift * ry / r * actual_dt
                
                # 5. Noise (Langevin)
                if noise_scale > 0:
                    x += rng.normal(0.0, noise_scale * np.sqrt(actual_dt), size=len(x))
                    y += rng.normal(0.0, noise_scale * np.sqrt(actual_dt), size=len(y))
                
                # 6. Cap cell count
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