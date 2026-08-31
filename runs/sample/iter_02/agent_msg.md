```python
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
            "Capsule-shaped cells (rod + two semicircular caps)",
            "Exponential elongation at constant rate, no crowding feedback",
            "Symmetric division at noisy target length (normal distribution)",
            "Overdamped position-based dynamics: soft repulsion and alignment torque",
            "Neighbor search via cKDTree with cutoff ~2.5 um",
            "Internal timestep ~15 s, 6 neighbor-search sweeps per frame",
            "No cell death or detachment",
            "Width fixed per cell, inherited at division",
        ],
        "state_vars": ["x","y","angle","length","width"],
        "refs": ["CellModeller-style agent-based model (Rudge et al. 2012)"],
    }

    def fit(self, obs):
        # Fit from FIT-window frames
        times = obs.time_s / 3600.0  # hours
        counts = np.array([len(f["x"]) for f in obs.frames])
        total_len = np.array([np.sum(f["length"]) for f in obs.frames])
        
        # Fit population growth rate (exponential)
        log2N = np.log2(counts)
        mask = times > 0.2 * times[-1]
        if np.sum(mask) > 2:
            slope, _ = np.polyfit(times[mask], log2N[mask], 1)
            doubling_time = 1.0 / slope
        else:
            doubling_time = 1.0
        
        elong_rate = np.log(2.0) / doubling_time
        
        # Estimate division length from length distribution
        all_lengths = np.concatenate([f["length"] for f in obs.frames])
        div_length = np.percentile(all_lengths, 90)
        
        # Division CV from high lengths
        high_len = all_lengths[all_lengths > np.percentile(all_lengths, 80)]
        if len(high_len) > 5:
            div_cv = np.std(high_len) / np.mean(high_len)
        else:
            div_cv = 0.1
        
        # Stiffness: heuristic based on initial density
        # Use mean NN distance to infer stiffness (higher stiffness -> less overlap)
        init_nn = self._mean_nn(obs.frames[0])
        if init_nn < 2.0:
            stiffness = 10.0
        elif init_nn < 3.0:
            stiffness = 5.0
        else:
            stiffness = 2.0
        
        return {
            "elong_rate": float(elong_rate),
            "div_length": float(div_length),
            "div_cv": float(np.clip(div_cv, 0.02, 0.35)),
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
        stiffness = params["stiffness"]
        
        dt = 15.0  # seconds
        cutoff = 2.5  # um
        n_sweeps = 3  # mechanical sweeps per step
        max_cells = 3000
        
        # Initialize
        x = init_cells["x"].copy().astype(np.float64)
        y = init_cells["y"].copy().astype(np.float64)
        angle = init_cells["angle"].copy().astype(np.float64)
        length = init_cells["length"].copy().astype(np.float64)
        width = init_cells["width"].copy().astype(np.float64)
        
        # Per-cell division threshold (drawn at birth)
        div_threshold = rng.normal(div_length, div_cv * div_length, size=len(x))
        div_threshold = np.maximum(div_threshold, 0.5)
        
        frames_out = []
        frames_out.append({
            "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
            "length": length.copy(), "width": width.copy()
        })
        
        t = time_s[0]
        frame_idx = 1
        next_frame_time = time_s[frame_idx] if frame_idx < len(time_s) else None
        
        while next_frame_time is not None:
            while t < next_frame_time - 1e-9:
                step = min(dt, next_frame_time - t)
                
                # Elongation
                length *= np.exp(elong_rate * step)
                
                # Division
                N = len(x)
                div_mask = length >= div_threshold
                if np.any(div_mask) and N < max_cells:
                    new_x, new_y, new_angle, new_length, new_width, new_div_thr = [], [], [], [], [], []
                    for i in range(N):
                        if div_mask[i]:
                            L = length[i] / 2.0
                            dx = L * 0.5 * np.cos(angle[i])
                            dy = L * 0.5 * np.sin(angle[i])
                            # Daughter 1
                            new_x.append(x[i] - dx)
                            new_y.append(y[i] - dy)
                            new_angle.append(angle[i])
                            new_length.append(L)
                            new_width.append(width[i])
                            new_div_thr.append(rng.normal(div_length, div_cv*div_length))
                            # Daughter 2
                            new_x.append(x[i] + dx)
                            new_y.append(y[i] + dy)
                            new_angle.append(angle[i])
                            new_length.append(L)
                            new_width.append(width[i])
                            new_div_thr.append(rng.normal(div_length, div_cv*div_length))
                        else:
                            new_x.append(x[i])
                            new_y.append(y[i])
                            new_angle.append(angle[i])
                            new_length.append(length[i])
                            new_width.append(width[i])
                            new_div_thr.append(div_threshold[i])
                    
                    x = np.array(new_x)
                    y = np.array(new_y)
                    angle = np.array(new_angle)
                    length = np.array(new_length)
                    width = np.array(new_width)
                    div_threshold = np.array(new_div_thr)
                    div_threshold = np.maximum(div_threshold, 0.5)
                
                # Mechanical relaxation
                for _ in range(n_sweeps):
                    self._mechanical_sweep(x, y, angle, length, width, stiffness, cutoff)
                
                t += step
                if t >= next_frame_time - 1e-9:
                    break
            
            frames_out.append({
                "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
                "length": length.copy(), "width": width.copy()
            })
            frame_idx += 1
            if frame_idx < len(time_s):
                next_frame_time = time_s[frame_idx]
            else:
                next_frame_time = None
        
        return frames_out

    def _mechanical_sweep(self, x, y, angle, length, width, stiffness, cutoff):
        N = len(x)
        if N == 0:
            return
        
        points = np.stack([x, y], axis=1)
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
        if len(pairs) == 0:
            return
        
        dx = x[pairs[:,0]] - x[pairs[:,1]]
        dy = y[pairs[:,0]] - y[pairs[:,1]]
        dist = np.sqrt(dx*dx + dy*dy)
        
        L_i = length[pairs[:,0]]
        L_j = length[pairs[:,1]]
        W_i = width[pairs[:,0]]
        W_j = width[pairs[:,1]]
        contact = (L_i + L_j) / 4.0 + (W_i + W_j) / 4.0
        overlap = contact - dist
        mask = overlap > 0
        
        if not np.any(mask):
            return
        
        idx_i = pairs[mask, 0]
        idx_j = pairs[mask, 1]
        dx_m = dx[mask]
        dy_m = dy[mask]
        dist_m = dist[mask]
        ov = overlap[mask]
        
        dist_safe = np.maximum(dist_m, 1e-6)
        ux = dx_m / dist_safe
        uy = dy_m / dist_safe
        
        # Position update: move proportional to overlap, capped
        move = np.clip(stiffness * ov * 0.05, 0, 0.5)
        
        np.add.at(x, idx_i, move * ux)
        np.add.at(y, idx_i, move * uy)
        np.subtract.at(x, idx_j, move * ux)
        np.subtract.at(y, idx_j, move * uy)
        
        # Alignment torque
        align_strength = 0.1
        d_angle = angle[idx_i] - angle[idx_j]
        torque = align_strength * ov * np.sin(d_angle)
        angle[idx_i] -= torque
        angle[idx_j] += torque
        angle = np.arctan2(np.sin(angle), np.cos(angle))
```