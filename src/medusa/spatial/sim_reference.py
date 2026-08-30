"""A simplified CellModeller-style rod-colony simulator.

Single source of truth: `SOURCE` is a complete, contract-compliant spatial `twin.py`
(numpy + scipy only). We also exec it here to expose `ReferenceTwin` for rendering,
tests, and as the canned dry-run twin. The agent's job is to produce something like
this -- or better.
"""

from __future__ import annotations

SOURCE = r'''
import numpy as np
from scipy.spatial import cKDTree


class Twin:
    """Overdamped rods on a surface: exponential elongation with crowding feedback,
    stochastic division, soft capsule-capsule repulsion (CellModeller-style)."""

    FAMILY = "overdamped-rods"

    PARAMS = {
        "elong_rate":   (0.2, 3.0, "1/h"),      # exponential length growth rate
        "div_length":   (4.0, 12.0, "um"),      # mean length at division
        "div_cv":       (0.02, 0.35, "fraction"),
        "n_half":       (200.0, 20000.0, "cells"),  # crowding: rate halves at this count
        "stiffness":    (0.5, 40.0, "um/(um*step)"),  # repulsion gain
        "align_rate":   (0.0, 6.0, "1/step"),   # how fast contacts torque rods apart
    }

    METADATA = {
        "assumptions": [
            "cells are rigid capsules of fixed width",
            "overdamped (inertteia-free) motion; velocity proportional to force",
            "exponential single-cell elongation, slowed by local crowding",
            "symmetric division at a noisy target length",
            "soft repulsion only (no adhesion, no chemotaxis, no nutrient field)",
        ],
        "state_vars": ["x", "y", "angle", "length"],
        "refs": ["Rudge et al. 2012 (CellModeller)", "Volfson et al. 2008"],
    }

    _DT_H = 0.04      # internal step (hours)
    _MAX_CELLS = 2500
    _RELAX_ITERS = 2  # position-based relaxation sweeps per step

    # ---- calibration -------------------------------------------------------

    def fit(self, obs):
        # obs.time_s : array[F]; obs.frames : list of {x,y,angle,length,width} (um)
        t_h = np.asarray(obs.time_s, float) / 3600.0
        counts = np.array([len(f["x"]) for f in obs.frames], float)
        biomass = np.array([np.sum(f["length"]) for f in obs.frames], float)
        lengths = np.concatenate([f["length"] for f in obs.frames])

        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])

        def _logslope(t, y):
            A = np.vstack([t, np.ones_like(t)]).T
            return float(np.linalg.lstsq(A, np.log(np.clip(y, 1e-9, None)), rcond=None)[0][0])

        # elongation rate: slope of log(total rod length) -- that IS the biomass rate
        elong = float(np.clip(_logslope(t_h, biomass), lo[0], hi[0]))

        div_len = float(np.clip(np.percentile(lengths, 90), lo[1], hi[1]))
        div_cv = float(np.clip(np.std(lengths) / max(np.mean(lengths), 1e-6), lo[2], hi[2]))

        # crowding only if the per-capita rate actually decays across the window
        h = len(t_h) // 2
        r_early = _logslope(t_h[:h], biomass[:h]) if h >= 3 else elong
        r_late = _logslope(t_h[h:], biomass[h:]) if len(t_h) - h >= 3 else elong
        if r_late < 0.7 * r_early and r_early > 0:
            n_half = float(np.clip(counts[h], lo[3], hi[3]))
        else:
            n_half = hi[3]

        return {
            "elong_rate": elong,
            "div_length": div_len,
            "div_cv": div_cv,
            "n_half": n_half,
            "stiffness": 8.0,
            "align_rate": 2.0,
        }

    # ---- rollout ----------------------------------------------------------

    def simulate(self, params, init_cells, time_s, seed):
        rng = np.random.default_rng(int(seed))
        x = np.asarray(init_cells["x"], float).copy()
        y = np.asarray(init_cells["y"], float).copy()
        ang = np.asarray(init_cells["angle"], float).copy()
        ln = np.asarray(init_cells["length"], float).copy()
        w = np.asarray(init_cells.get("width", np.ones_like(x)), float)
        width = float(np.median(w)) if w.size else 1.0

        k = float(params["elong_rate"])
        Ld = float(params["div_length"])
        cv = float(params["div_cv"])
        n_half = float(params["n_half"])
        stiff = float(params["stiffness"])
        align = float(params["align_rate"])

        def _new_targets(m):
            return np.maximum(Ld * (1.0 + cv * rng.standard_normal(m)), 0.55 * Ld)

        div_target = _new_targets(x.size)
        len_cap = 1.6 * Ld
        times = np.asarray(time_s, float) / 3600.0   # contract passes seconds; work in hours
        t = float(times[0])
        out = [self._frame(x, y, ang, ln, width)]

        for tnext in times[1:]:
            while t < tnext - 1e-9:
                dt = min(self._DT_H, tnext - t)
                rate = k / (1.0 + x.size / n_half)
                ln = np.minimum(ln * np.exp(rate * dt), len_cap)

                div = np.where(ln >= div_target)[0]
                if div.size and x.size < self._MAX_CELLS:
                    x, y, ang, ln, div_target = self._divide(
                        x, y, ang, ln, div_target, div, _new_targets, rng
                    )

                if x.size > 1:
                    for _ in range(self._RELAX_ITERS):
                        x, y, ang = self._relax(x, y, ang, ln, width, stiff, align, dt)
                t += dt
            out.append(self._frame(x, y, ang, ln, width))
        return out

    @staticmethod
    def _frame(x, y, ang, ln, width):
        return {
            "x": x.copy(), "y": y.copy(), "angle": ang.copy(),
            "length": ln.copy(), "width": np.full(x.size, width),
        }

    @staticmethod
    def _divide(x, y, ang, ln, div_target, idx, new_targets, rng):
        ux, uy = np.cos(ang[idx]), np.sin(ang[idx])
        child = ln[idx] / 2.0
        off = child / 2.0

        x2, y2 = x[idx] - ux * off, y[idx] - uy * off
        x[idx], y[idx], ln[idx] = x[idx] + ux * off, y[idx] + uy * off, child
        ang[idx] = ang[idx] + 0.03 * rng.standard_normal(idx.size)
        div_target[idx] = new_targets(idx.size)

        x = np.concatenate([x, x2])
        y = np.concatenate([y, y2])
        ang = np.concatenate([ang, ang[idx] + 0.03 * rng.standard_normal(idx.size)])
        ln = np.concatenate([ln, child])
        div_target = np.concatenate([div_target, new_targets(idx.size)])
        return x, y, ang, ln, div_target

    @staticmethod
    def _relax(x, y, ang, ln, width, stiff, align, dt):
        pts = np.column_stack([x, y])
        cutoff = float(min(ln.max(), 3.0 * np.median(ln)) + width)
        pairs = cKDTree(pts).query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            return x, y, ang
        i, j = pairs[:, 0], pairs[:, 1]

        ci, cj = _closest_on_axes(x[i], y[i], ang[i], ln[i], x[j], y[j], ang[j], ln[j])
        dx = ci[0] - cj[0]
        dy = ci[1] - cj[1]
        dist = np.hypot(dx, dy)
        contact = width
        overlap = contact - dist
        m = overlap > 0
        if not np.any(m):
            return x, y, ang

        nx = np.where(dist > 1e-9, dx / np.maximum(dist, 1e-9), 1.0)[m]
        ny = np.where(dist > 1e-9, dy / np.maximum(dist, 1e-9), 0.0)[m]
        f = stiff * overlap[m]
        ii, jj = i[m], j[m]

        fx = np.zeros_like(x)
        fy = np.zeros_like(y)
        np.add.at(fx, ii, f * nx)
        np.add.at(fy, ii, f * ny)
        np.add.at(fx, jj, -f * nx)
        np.add.at(fy, jj, -f * ny)

        x = x + fx * dt
        y = y + fy * dt

        # torque: rotate rods so their long axis turns away from the contact normal
        ux, uy = np.cos(ang), np.sin(ang)
        tq = np.zeros_like(ang)
        cross_i = (ci[0][m] - x[ii]) * (f * ny) - (ci[1][m] - y[ii]) * (f * nx)
        cross_j = (cj[0][m] - x[jj]) * (-f * ny) - (cj[1][m] - y[jj]) * (-f * nx)
        np.add.at(tq, ii, cross_i)
        np.add.at(tq, jj, cross_j)
        ang = ang + align * tq * dt / np.maximum(ln, 1e-6)
        return x, y, ang

    def predict(self, params, time_s):  # convenience for series-style probing
        raise NotImplementedError("spatial twin: use simulate()")


def _closest_on_axes(xi, yi, ai, li, xj, yj, aj, lj):
    """Closest points between two rod centre-axis segments (vectorised, approximate)."""
    ui = np.column_stack([np.cos(ai), np.sin(ai)])
    uj = np.column_stack([np.cos(aj), np.sin(aj)])
    pi = np.column_stack([xi, yi])
    pj = np.column_stack([xj, yj])
    r = pi - pj
    a = np.einsum("ij,ij->i", ui, ui) * (li / 2) ** 2
    e = np.einsum("ij,ij->i", uj, uj) * (lj / 2) ** 2
    b = np.einsum("ij,ij->i", ui, uj) * (li / 2) * (lj / 2)
    c = np.einsum("ij,ij->i", ui, r) * (li / 2)
    fd = np.einsum("ij,ij->i", uj, r) * (lj / 2)
    denom = a * e - b * b
    s = np.where(np.abs(denom) > 1e-9, (b * fd - c * e) / np.where(denom == 0, 1, denom), 0.0)
    s = np.clip(s, -1.0, 1.0)
    tt = np.clip((b * s + fd) / np.where(e == 0, 1, e), -1.0, 1.0)
    s = np.clip((b * tt - c) / np.where(a == 0, 1, a), -1.0, 1.0)
    ci = pi + ui * (s * (li / 2))[:, None]
    cj = pj + uj * (tt * (lj / 2))[:, None]
    return (ci[:, 0], ci[:, 1]), (cj[:, 0], cj[:, 1])
'''

_ns: dict = {}
exec(compile(SOURCE, "sim_reference_twin.py", "exec"), _ns)  # noqa: S102
ReferenceTwin = _ns["Twin"]
