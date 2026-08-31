```python
import numpy as np
from scipy.spatial import cKDTree

class Twin:
    FAMILY = "continuum-density-orientation"
    PARAMS = {
        "elong_rate": (0.2, 3.0, "1/h"),
        "div_length": (3.0, 12.0, "um"),
        "div_cv":     (0.02, 0.35, "fraction"),
        "pressure_coupling": (0.0, 2.0, "um^2"),
    }
    METADATA = {
        "assumptions": [
            "Continuum density field with local growth, pressure-driven expansion",
            "Orientation field with nematic alignment (Maier-Saupe type)",
            "Cell length distribution follows a gamma with mean div_length/2 and CV ~ div_cv",
            "Emit representative cell positions from a Poisson hard-core process matched to local density",
            "No individual cell mechanics; all interactions captured by pressure and alignment",
            "Colony expands isotropically with pressure; shape kept roughly elliptical",
            "Width fixed at 1.0 um for all cells",
            "No death, no detachment",
        ],
        "state_vars": ["x","y","angle","length","width"],
        "refs": ["Continuum models of microbial colonies (e.g. Farrell et al. 2013)"],
    }

    def fit(self, obs):
        # Fit from FIT-window frames
        times = obs.time_s / 3600.0  # hours
        counts = np.array([len(f["x"]) for f in obs.frames])
        total_len = np.array([np.sum(f["length"]) for f in obs.frames])
        
        # Population growth rate (exponential fit)
        log2N = np.log2(counts)
        mask = times > 0.2 * times[-1]
        if np.sum(mask) > 2:
            slope, _ = np.polyfit(times[mask], log2N[mask], 1)
            doubling_time = 1.0 / slope
        else:
            doubling_time = 1.0
        elong_rate = np.log(2.0) / doubling_time
        
        # Division length from length distribution (high percentile)
        all_lengths = np.concatenate([f["length"] for f in obs.frames])
        div_length = np.percentile(all_lengths, 90)
        
        # CV of division length (from high lengths)
        high_len = all_lengths[all_lengths > np.percentile(all_lengths, 80)]
        if len(high_len) > 5:
            div_cv = np.std(high_len) / np.mean(high_len)
        else:
            div_cv = 0.1
        
        # Pressure coupling: from initial density (mean NN distance)
        init_nn = self._mean_nn(obs.frames[0])
        # Higher density -> more pressure -> weaker coupling (smaller value)
        pressure_coupling = 0.5 + 2.0 * (1.0 - np.clip(init_nn / 3.0, 0, 1)) ** 2
        
        return {
            "elong_rate": float(elong_rate),
            "div_length": float(div_length),
            "div_cv": float(np.clip(div_cv, 0.02, 0.35)),
            "pressure_coupling": float(np.clip(pressure_coupling, 0.0, 2.0)),
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
        pressure_coupling = params["pressure_coupling"]
        
        dt = 30.0  # seconds (coarser than agent-based)
        n_steps_per_frame = max(1, int(round((time_s[1] - time_s[0]) / dt)))
        
        # Initialize from first frame
        x0 = init_cells["x"].copy().astype(np.float64)
        y0 = init_cells["y"].copy().astype(np.float64)
        angle0 = init_cells["angle"].copy().astype(np.float64)
        length0 = init_cells["length"].copy().astype(np.float64)
        width0 = init_cells["width"].copy().astype(np.float64)
        
        # Compute initial colony statistics for continuum field
        N0 = len(x0)
        total_len0 = np.sum(length0)
        # Center of mass
        cx0 = np.mean(x0)
        cy0 = np.mean(y0)
        # Radius of gyration (RMS distance from center)
        r0 = np.sqrt(np.mean((x0 - cx0)**2 + (y0 - cy0)**2))
        # Aspect ratio from covariance
        cov = np.cov(np.stack([x0, y0]))
        eigvals = np.linalg.eigvalsh(cov)
        aspect0 = np.sqrt(eigvals[1] / max(eigvals[0], 1e-12)) if eigvals[0] > 1e-12 else 1.0
        
        # Nematic order from initial angles
        order0 = np.abs(np.mean(np.exp(2j * angle0)))
        
        frames_out = []
        frames_out.append({
            "x": x0.copy(), "y": y0.copy(), "angle": angle0.copy(),
            "length": length0.copy(), "width": width0.copy()
        })
        
        # State variables: N (count), total_length, radius, aspect, order
        N = N0
        total_len = total_len0
        radius = r0
        aspect = aspect0
        order = order0
        center = np.array([cx0, cy0])
        
        # Seed for generating cell configurations
        rng_gen = np.random.default_rng(seed + 12345)
        
        for frame_idx in range(1, len(time_s)):
            # Time advance over this frame interval
            dt_frame = time_s[frame_idx] - time_s[frame_idx-1]
            n_steps = max(1, int(round(dt_frame / dt)))
            
            for _ in range(n_steps):
                # 1. Growth (exponential, with pressure feedback)
                # Pressure: local density ~ N / (pi * radius^2)
                area = np.pi * radius**2
                density = N / max(area, 1.0)
                # Pressure reduces growth (logistic-like): growth_rate = base * (1 - coupling*density)
                growth_factor = max(0.0, 1.0 - pressure_coupling * density)
                dN = elong_rate * N * growth_factor * dt
                N += dN
                # Length increase proportional to N increase (each new cell adds length ~ L/2)
                avg_len = total_len / max(N, 1)
                total_len += avg_len * dN * 0.5  # each birth adds half length? Actually total length grows with N and avg length
                
                # 2. Radius expansion: driven by growth and pressure
                # Simple: radius grows with sqrt(N) (area ~ N)
                radius = max(radius, np.sqrt(N / np.pi) * 1.0)  # ensure at least that
                # Also add a small expansion from pressure (stiffness-like)
                radius += 0.01 * growth_factor * radius * dt / 3600.0
                
                # 3. Aspect ratio: slowly relax toward 1 (isotropic) but keep memory
                aspect = aspect + 0.01 * (1.0 - aspect) * dt / 3600.0
                
                # 4. Nematic order: relax toward 0.2 (typical) with noise
                order = order + 0.05 * (0.2 - order) * dt / 3600.0
                order = np.clip(order, 0.0, 1.0)
            
            # Generate a cell configuration consistent with these summary stats
            N_int = int(round(N))
            N_int = min(N_int, 3000)
            
            # Place cells in an elliptical annulus with density proportional to local density
            # Use a simple method: sample from a 2D Gaussian with covariance matching aspect
            # Then hard-core rejection to achieve mean NN distance ~ 1/sqrt(density)
            target_nn = 1.0 / np.sqrt(N_int / (np.pi * radius**2))
            target_nn = np.clip(target_nn, 0.5, 3.0)
            
            # Generate candidate positions
            cov_xy = np.array([[radius**2 / 2.0, 0], [0, radius**2 / 2.0]])
            # Scale to match aspect ratio
            cov_xy[0,0] *= aspect
            cov_xy[1,1] /= aspect
            # Generate many candidates, then thin to N_int with hard-core
            n_candidates = int(N_int * 5)
            candidates = rng_gen.multivariate_normal(center, cov_xy, size=n_candidates)
            # Keep only within radius*1.5
            dist_from_center = np.linalg.norm(candidates - center, axis=1)
            candidates = candidates[dist_from_center < radius * 1.5]
            
            # Hard-core thinning: greedy selection with minimum distance ~ target_nn * 0.7
            if len(candidates) > 0:
                selected = []
                tree = None
                for pt in candidates:
                    if len(selected) == 0:
                        selected.append(pt)
                        tree = cKDTree(np.array(selected))
                    else:
                        d, _ = tree.query(pt)
                        if d > target_nn * 0.7:
                            selected.append(pt)
                            tree = cKDTree(np.array(selected))
                    if len(selected) >= N_int:
                        break
                if len(selected) < N_int:
                    # Fill with random points (some overlap allowed)
                    while len(selected) < N_int:
                        pt = rng_gen.multivariate_normal(center, cov_xy)
                        selected.append(pt)
                selected = np.array(selected[:N_int])
            else:
                # Fallback: just random points
                selected = rng_gen.multivariate_normal(center, cov_xy, size=N_int)
            
            x = selected[:, 0]
            y = selected[:, 1]
            
            # Assign lengths: gamma distribution with mean div_length/2 (since cells divide at div_length)
            mean_len = div_length / 2.0
            # Use gamma with shape from CV
            shape = 1.0 / (div_cv**2)
            scale = mean_len / shape
            lengths = rng_gen.gamma(shape, scale, size=N_int)
            lengths = np.clip(lengths, 0.5, div_length * 1.2)
            
            # Assign angles: nematic distribution with given order parameter
            # Sample from von Mises with concentration kappa such that order ~ <cos(2theta)>
            # Approximate: kappa ~ 2*order / (1-order) for small order
            kappa = max(0.0, 2.0 * order / max(1.0 - order, 1e-6))
            angles = rng_gen.vonmises(0.0, kappa, size=N_int)
            # Add a global orientation (use initial mean angle)
            mean_angle = np.arctan2(np.mean(np.sin(angle0)), np.mean(np.cos(angle0)))
            angles += mean_angle
            
            # Width fixed
            widths = np.full(N_int, 1.0)
            
            frames_out.append({
                "x": x, "y": y, "angle": angles,
                "length": lengths, "width": widths
            })
        
        return frames_out
```