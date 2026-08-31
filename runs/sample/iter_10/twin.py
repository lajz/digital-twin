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
            "Cells are rod-shaped capsules with fixed width ~1 um",
            "Exponential elongation with no density feedback (observed growth is close to exponential)",
            "Symmetric division at noisy target length, daughters inherit parent orientation",
            "Soft repulsion between overlapping capsules, position-based dynamics with multiple sweeps",
            "Nematic alignment via torque on contact: rods align parallel to each other when in contact",
            "No cell death or detachment",
            "Neighbour search via cKDTree with adaptive cutoff",
            "Internal timestep set to ~90 s (frame interval) for speed, with substeps only for division events",
            "Small positional noise to mimic segmentation error",
            "Cell count capped at 3000",
        ],
        "state_vars": ["x","y","angle","length","width"],
        "refs": [
            "Rudge et al. 2012 (CellModeller)",
            "Ahmadi et al. 2024 (IPB benchmark)",
        ],
    }

    def fit(self, obs):
        times = obs.time_s / 3600.0
        counts = np.array([len(f["x"]) for f in obs.frames])
        total_len = np.array([np.sum(f["length"]) for f in obs.frames])
        radii = np.array([self._gyration_radius(f) for f in obs.frames])
        nematic = np.array([self._nematic_order(f) for f in obs.frames])
        nn_dist = np.array([self._mean_nn(f) for f in obs.frames])

        # Fit exponential growth rate from population count (log2)
        log2N = np.log2(np.maximum(counts, 1))
        mask = times > 0.1 * times[-1]
        if np.sum(mask) > 2:
            slope, _ = np.polyfit(times[mask], log2N[mask], 1)
            doubling_time = 1.0 / max(slope, 1e-6)
        else:
            doubling_time = 0.5
        elong_rate = np.log(2.0) / doubling_time

        # Division length: 85th percentile of observed lengths (late exponential phase)
        all_lengths = np.concatenate([f["length"] for f in obs.frames])
        div_length = np.percentile(all_lengths, 85)

        # CV from lengths above 70th percentile (those are near division)
        high_len = all_lengths[all_lengths > np.percentile(all_lengths, 70)]
        if len(high_len) > 5:
            div_cv = np.std(high_len) / np.mean(high_len)
        else:
            div_cv = 0.1
        div_cv = np.clip(div_cv, 0.02, 0.35)

        # Stiffness from initial NN distance: more crowded => higher stiffness needed
        init_nn = nn_dist[0] if len(nn_dist) > 0 else 2.5
        stiffness = 5.0 + 15.0 * (2.5 - init_nn)
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

        elong_rate = params["elong_rate"] / 3600.0  # per sec
        div_length = params["div_length"]
        div_cv = params["div_cv"]
        stiffness = params["stiffness"]

        width = 1.0  # um, typical E. coli width

        # Use frame interval as internal dt (90 s) for speed, but substep for division
        dt = 90.0  # seconds per internal step

        x = init_cells["x"].copy().astype(np.float64)
        y = init_cells["y"].copy().astype(np.float64)
        angle = init_cells["angle"].copy().astype(np.float64)
        length = init_cells["length"].copy().astype(np.float64)
        w = np.full(len(x), width)

        # Division targets (noisy around div_length)
        div_target = div_length * (1.0 + rng.normal(0.0, div_cv, size=len(x)))

        # Output list
        frames_out = []
        frames_out.append({
            "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
            "length": length.copy(), "width": w.copy()
        })

        current_time = time_s[0]
        frame_idx = 1

        while frame_idx < len(time_s):
            next_time = time_s[frame_idx]
            dt_frame = next_time - current_time
            # Use frame interval as one step, but if frame interval differs, use that
            n_steps = max(1, int(round(dt_frame / dt)))
            actual_dt = dt_frame / n_steps

            for step in range(n_steps):
                # Elongation (exponential, no density feedback)
                length *= np.exp(elong_rate * actual_dt)

                # Division: split cells that reached target length
                if len(x) > 0:
                    mask_div = length >= div_target
                    if np.any(mask_div):
                        idx_div = np.where(mask_div)[0]
                        n_div = len(idx_div)
                        new_x = np.empty(n_div)
                        new_y = np.empty(n_div)
                        new_angle = np.empty(n_div)
                        new_length = np.empty(n_div)
                        new_w = np.full(n_div, width)

                        for i, idx in enumerate(idx_div):
                            L = length[idx]
                            half = L / 2.0
                            # Place daughters at parent position, push apart later
                            new_x[i] = x[idx]
                            new_y[i] = y[idx]
                            new_angle[i] = angle[idx]
                            new_length[i] = half
                            length[idx] = half
                            # Reset target for parent
                            div_target[idx] = div_length * (1.0 + rng.normal(0.0, div_cv))

                        x = np.concatenate([x, new_x])
                        y = np.concatenate([y, new_y])
                        angle = np.concatenate([angle, new_angle])
                        length = np.concatenate([length, new_length])
                        w = np.concatenate([w, new_w])
                        div_target = np.concatenate([div_target,
                                                     div_length * (1.0 + rng.normal(0.0, div_cv, size=n_div))])

                # Mechanical relaxation: multiple position-based sweeps for stability
                if len(x) > 1:
                    # Neighbour search with adaptive cutoff
                    max_L = np.max(length)
                    cutoff = 2.0 * max_L + 1.0
                    pts = np.stack([x, y], axis=1)
                    tree = cKDTree(pts)
                    pairs = tree.query_pairs(cutoff, output_type='ndarray')

                    # Do a few relaxation sweeps (2 is enough for stability)
                    for sweep in range(2):
                        if len(pairs) > 0:
                            fx = np.zeros(len(x))
                            fy = np.zeros(len(x))
                            torque = np.zeros(len(x))

                            i_idx = pairs[:, 0]
                            j_idx = pairs[:, 1]
                            dx = x[j_idx] - x[i_idx]
                            dy = y[j_idx] - y[i_idx]
                            dist = np.hypot(dx, dy)

                            valid = dist > 1e-6
                            if np.any(valid):
                                i_v = i_idx[valid]
                                j_v = j_idx[valid]
                                dx_v = dx[valid]
                                dy_v = dy[valid]
                                dist_v = dist[valid]

                                hi = length[i_v] / 2.0
                                hj = length[j_v] / 2.0

                                cos_i = np.cos(angle[i_v])
                                sin_i = np.sin(angle[i_v])
                                cos_j = np.cos(angle[j_v])
                                sin_j = np.sin(angle[j_v])

                                # Project centres onto rod axes
                                proj_i = dx_v * cos_i + dy_v * sin_i
                                proj_j = dx_v * cos_j + dy_v * sin_j

                                proj_i_c = np.clip(proj_i, -hi, hi)
                                proj_j_c = np.clip(proj_j, -hj, hj)

                                # Closest points on each rod
                                cx_i = x[i_v] + cos_i * proj_i_c
                                cy_i = y[i_v] + sin_i * proj_i_c
                                cx_j = x[j_v] + cos_j * proj_j_c
                                cy_j = y[j_v] + sin_j * proj_j_c

                                dist_c = np.hypot(cx_j - cx_i, cy_j - cy_i)
                                valid2 = dist_c > 1e-6

                                if np.any(valid2):
                                    i_v2 = i_v[valid2]
                                    j_v2 = j_v[valid2]
                                    dist_c2 = dist_c[valid2]
                                    cx_i2 = cx_i[valid2]
                                    cy_i2 = cy_i[valid2]
                                    cx_j2 = cx_j[valid2]
                                    cy_j2 = cy_j[valid2]

                                    overlap = 1.0 - dist_c2  # width ~1 um
                                    pos_overlap = overlap > 0

                                    if np.any(pos_overlap):
                                        i_v3 = i_v2[pos_overlap]
                                        j_v3 = j_v2[pos_overlap]
                                        ov = overlap[pos_overlap]
                                        cx_i3 = cx_i2[pos_overlap]
                                        cy_i3 = cy_i2[pos_overlap]
                                        cx_j3 = cx_j2[pos_overlap]
                                        cy_j3 = cy_j2[pos_overlap]
                                        dist_c3 = dist_c2[pos_overlap]

                                        # Force magnitude (scaled by stiffness and overlap)
                                        force_mag = stiffness * ov

                                        nx = (cx_j3 - cx_i3) / dist_c3
                                        ny = (cy_j3 - cy_i3) / dist_c3

                                        np.add.at(fx, i_v3, force_mag * nx)
                                        np.add.at(fy, i_v3, force_mag * ny)
                                        np.add.at(fx, j_v3, -force_mag * nx)
                                        np.add.at(fy, j_v3, -force_mag * ny)

                                        # Nematic alignment torque: align rods to contact direction
                                        # (simplified: align each rod to the other's orientation)
                                        diff_i = angle[j_v3] - angle[i_v3]
                                        diff_i = (diff_i + np.pi/2) % np.pi - np.pi/2
                                        torque_i = force_mag * 0.5 * np.sin(2*diff_i)
                                        np.add.at(torque, i_v3, torque_i)
                                        np.add.at(torque, j_v3, -torque_i)

                        # Apply forces and torques (overdamped)
                        x += fx * actual_dt * 0.5  # half step for stability
                        y += fy * actual_dt * 0.5
                        angle += torque * actual_dt * 0.5

                # Small noise to mimic segmentation error and stochasticity
                noise_scale = 0.02 * actual_dt  # ~0.02 um/s
                x += rng.normal(0.0, noise_scale, size=len(x))
                y += rng.normal(0.0, noise_scale, size=len(y))

                # Cap cell count
                if len(x) > 3000:
                    x = x[:3000]
                    y = y[:3000]
                    angle = angle[:3000]
                    length = length[:3000]
                    w = w[:3000]
                    div_target = div_target[:3000]

            frames_out.append({
                "x": x.copy(), "y": y.copy(), "angle": angle.copy(),
                "length": length.copy(), "width": w.copy()
            })

            current_time = next_time
            frame_idx += 1

        return frames_out