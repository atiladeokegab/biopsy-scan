"""Batched fused GPU re-imaging: many studies per launch, from the random stream to z, on the device.

One CUDA thread per look (one region of one drawn patient of one study) runs reimage.fused._region's
loop in the same order: it steps NumPy's PCG64 stream for the per-cell uniforms, mis-calls, crops, counts
neighbours and sums fractions in increasing cell index. A second kernel pools looks into patient
features and stats.gpu_score scores every study. So X is bit-identical to the CPU fused arm: the same
IEEE operations in the same order, and no multiply-add pair the compiler could contract into an FMA.

The host keeps only what needs NumPy's Generator: the patient and region draws and each look's two crop
uniforms (plan). It never draws a per-cell uniform: it jumps the stream past them and hands the device
each look's PCG64 state. backend="cpu" runs the same loop bodies under numba.njit (the reference path).
"""
from time import perf_counter

import numpy as np
from numba import cuda, njit, prange

from core.reimage_ref import confusion_matrix
from core.truth import R, layout, neighbours
from ._common import count_inputs

KMAX = 127  # observed types are stored as int8, -1 = not visible
MULT = 0x2360ED051FC65DA44385DF649FCCF645  # PCG64's 128-bit LCG multiplier (numpy/random/src/pcg64)
M64, M128 = (1 << 64) - 1, (1 << 128) - 1
_MHI, _MLO = np.uint64(MULT >> 64), np.uint64(MULT & M64)
_L1, _L0 = np.uint64((MULT & M64) >> 32), np.uint64(MULT & 0xFFFFFFFF)
_M32, _S32, _S58, _S11, _S63, _S64, _ONE = (np.uint64(v) for v in (0xFFFFFFFF, 32, 58, 11, 63, 64, 1))
_INV53 = 1.0 / 9007199254740992.0
_NAN = float(np.array([0xFFF8000000000000], np.uint64).view(np.float64)[0])  # x86's 0.0 / 0.0, the CPU arm's NaN bits


def _out(s: int) -> float:
    """PCG64's XSL-RR output of a 128-bit state, as Generator.random's double."""
    x, rot = (s >> 64) ^ (s & M64), s >> 122
    return ((((x >> rot) | (x << (64 - rot))) & M64) >> 11) * _INV53


def static(cells: dict, design: dict) -> dict:
    """Per-cohort constants: region boxes, PCG64 jumps over each region's uniforms, the mis-call table."""
    L = layout(cells)
    coff = np.searchsorted(np.asarray(cells["region_id"]), np.arange(len(cells["patient_of_region"]) + 1))
    nreg, fov = len(coff) - 1, design.get("fov_um")
    if L["K"] > KMAX:
        raise ValueError(f"at most {KMAX} cell types")
    lo, w, nocrop = np.zeros((nreg, 2)), np.zeros((nreg, 2)), np.ones(nreg, bool)
    jump = []
    for r in range(nreg):
        n = int(coff[r + 1] - coff[r])
        g = 0
        for _ in range(n):  # G_n = 1 + MULT + ... + MULT^(n-1): n steps map s to MULT^n s + G_n inc
            g = (g * MULT + 1) & M128
        jump.append((pow(MULT, n, 1 << 128), g))
        if fov is not None and n:  # core.reimage_ref.random_crop's own expressions, per region
            xy = L["xy"](r)
            a, b = xy.min(0), xy.max(0)
            nocrop[r] = float(fov) >= (b - a).max()
            lo[r], w[r] = a, np.maximum(b - a - float(fov), 0)
    t_of = np.full(L["K"], -1, np.int64)
    t_of[L["focal"]] = np.arange(len(L["focal"]))
    return {"L": L, "coff": coff, "lo": lo, "w": w, "nocrop": nocrop, "jump": jump, "fov": fov, "t_of": t_of,
            "design": dict(design), "Mcum": np.cumsum(confusion_matrix(L["K"], float(design.get("eps") or 0.0)), axis=1)}


def plan(st: dict, seeds) -> dict:
    """Every study's draws in core.reimage_ref.draw's order, minus the per-cell uniforms (the device makes those)."""
    L, d, fov, jump = st["L"], st["design"], st["fov"], st["jump"]
    pats, k = d.get("patients"), d.get("regions_per_patient")
    reg, row, a, state, inc, patients = [], [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = (np.arange(L["P"]) if pats is None else rng.integers(0, L["P"], int(pats)) if np.ndim(pats) == 0
               else np.asarray(pats, np.int64))
        regs = []
        for p in idx:
            rs = L["regions_of"][p]
            regs.append(rng.choice(rs, k, replace=False) if k is not None and k < len(rs) else rs)
        g = rng.bit_generator.state["state"]  # random() never reads the buffered uint32, so the state is enough
        s, c = g["state"], g["inc"]
        base = len(patients) * len(idx)
        for i, rs in enumerate(regs):
            for r in rs:
                if fov is not None:  # random_crop always draws its two uniforms
                    s = (s * MULT + c) & M128
                    a.append(_out(s))
                    s = (s * MULT + c) & M128
                    a.append(_out(s))
                reg.append(int(r))
                row.append(base + i)
                state.append(s)
                inc.append(c)
                A, G = jump[r]
                s = (A * s + G * c) & M128
        patients.append(idx)
    u64 = lambda v, shift: np.array([(x >> shift) & M64 for x in v], np.uint64)  # noqa: E731
    return {"patients": np.array(patients, np.int64).reshape(len(patients), -1), "reg": np.array(reg, np.int64),
            "row": np.array(row, np.int64), "a": np.array(a).reshape(-1, 2) if fov is not None else None,
            "hi": u64(state, 64), "lo": u64(state, 0), "ihi": u64(inc, 64), "ilo": u64(inc, 0)}


def merge(parts: list) -> dict:
    """Plans of consecutive seed chunks as one batch (patient rows renumbered)."""
    out = {k: (None if parts[0][k] is None else np.concatenate([p[k] for p in parts])) for k in parts[0] if k != "row"}
    rows = np.cumsum([0] + [p["patients"].size for p in parts[:-1]])
    out["row"] = np.concatenate([p["row"] + o for p, o in zip(parts, rows)])
    return out


def _look(l, reg, coff, typ, x, y, estart, ends, idx, hi, lo, ihi, ilo, box, ooff, Mcum, t_of, obs, p, cnt, Sl, ml):
    """reimage.fused._region for look l, drawing its uniforms from the look's PCG64 state."""
    K = Mcum.shape[0]
    c0 = coff[reg[l]]
    n = coff[reg[l] + 1] - c0
    o0 = ooff[l]
    sh, sl, ih, il = hi[l], lo[l], ihi[l], ilo[l]
    for k in range(K):
        p[l, k] = 0.0
    for c in range(Sl.shape[1]):
        Sl[l, c] = 0.0
    for t in range(ml.shape[1]):
        ml[l, t] = 0
    nvis = 0
    for i in range(n):
        a0, a1 = sl & _M32, sl >> _S32  # state = state * MULT + inc (mod 2^128), in 64-bit limbs
        p00, p01, p10 = a0 * _L0, a0 * _L1, a1 * _L0
        mid = (p00 >> _S32) + (p01 & _M32) + (p10 & _M32)
        nh = a1 * _L1 + (p01 >> _S32) + (p10 >> _S32) + (mid >> _S32) + sh * _MLO + sl * _MHI + ih
        sl = sl * _MLO + il
        if sl < il:
            nh += _ONE
        sh = nh
        xr, rot = sh ^ sl, sh >> _S58
        u = float(((xr >> rot) | (xr << ((_S64 - rot) & _S63))) >> _S11) * _INV53
        g = c0 + i
        if x[g] >= box[l, 0] and x[g] < box[l, 1] and y[g] >= box[l, 2] and y[g] < box[l, 3]:
            c = 0
            for k in range(K):  # core.reimage_ref.miscall: first k with cumsum(M[type])[k] > u
                if Mcum[typ[g], k] <= u:
                    c += 1
            if c > K - 1:
                c = K - 1
            obs[o0 + i] = c
            p[l, c] += 1.0
            nvis += 1
        else:
            obs[o0 + i] = -1
    if nvis == 0:
        return
    for k in range(K):
        p[l, k] = p[l, k] / float(nvis)
    for i in range(n):
        if obs[o0 + i] < 0:
            continue
        t = t_of[obs[o0 + i]]
        if t < 0:
            continue
        g = c0 + i
        e = estart[g]
        for r in range(R):  # row i's neighbours within radius r but not r - 1, then cumulate: exact integers
            for k in range(K):
                cnt[l, r * K + k] = cnt[l, (r - 1) * K + k] if r > 0 else 0
            while e < ends[g, r]:
                oj = obs[o0 + idx[e]]
                if oj >= 0:
                    cnt[l, r * K + oj] += 1
                e += 1
        ml[l, t] += 1
        for r in range(R):
            tot = 0
            for k in range(K):
                tot += cnt[l, r * K + k]
            for k in range(K):  # a zero count adds 0.0 on the CPU: skipping it leaves the same bits
                if cnt[l, r * K + k] > 0:
                    frac = cnt[l, r * K + k] / float(tot)
                    col = ((t * K + k) * R + r) * 2
                    Sl[l, col] += frac
                    if p[l, k] > 0:
                        Sl[l, col + 1] += frac / p[l, k]


def _patient(row, rptr, Sl, ml, X):
    """core.grammar.features for one drawn patient: its looks summed in draw order, then the focal-cell mean."""
    T = ml.shape[1]
    per = Sl.shape[1] // T
    s, i = row // X.shape[1], row % X.shape[1]
    for t in range(T):
        m = 0
        for l in range(rptr[row], rptr[row + 1]):
            m += ml[l, t]
        for c in range(t * per, (t + 1) * per):
            acc = 0.0
            for l in range(rptr[row], rptr[row + 1]):
                acc += Sl[l, c]
            X[s, i, c] = acc / float(m) if m > 0 else _NAN


_look_cpu, _patient_cpu = njit(cache=True)(_look), njit(cache=True)(_patient)
_look_dev, _patient_dev = cuda.jit(device=True)(_look), cuda.jit(device=True)(_patient)


@njit(parallel=True, cache=True)
def _run_cpu(reg, coff, typ, x, y, estart, ends, idx, hi, lo, ihi, ilo, box, ooff, Mcum, t_of, obs, p, cnt, Sl, ml,
             rptr, X):
    for l in prange(len(reg)):
        _look_cpu(l, reg, coff, typ, x, y, estart, ends, idx, hi, lo, ihi, ilo, box, ooff, Mcum, t_of, obs, p, cnt,
                  Sl, ml)
    for row in prange(len(rptr) - 1):
        _patient_cpu(row, rptr, Sl, ml, X)


@cuda.jit
def _looks_kernel(reg, coff, typ, x, y, estart, ends, idx, hi, lo, ihi, ilo, box, ooff, Mcum, t_of, obs, p, cnt, Sl,
                  ml):
    l = cuda.grid(1)
    if l < reg.shape[0]:
        _look_dev(l, reg, coff, typ, x, y, estart, ends, idx, hi, lo, ihi, ilo, box, ooff, Mcum, t_of, obs, p, cnt,
                  Sl, ml)


@cuda.jit
def _patients_kernel(rptr, Sl, ml, X):
    row = cuda.grid(1)
    if row < rptr.shape[0] - 1:
        _patient_dev(row, rptr, Sl, ml, X)


class Fused:
    """A cohort's adjacency, resident on the device (backend="gpu") or in host memory (backend="cpu")."""

    def __init__(self, cells: dict, design: dict, backend: str = "gpu", st: dict | None = None):
        if backend not in ("gpu", "cpu"):
            raise ValueError("backend must be 'gpu' or 'cpu'")
        if backend == "gpu" and not cuda.is_available():
            raise RuntimeError("backend='gpu' needs a CUDA device (or NUMBA_ENABLE_CUDASIM=1 for the simulator)")
        self.st, self.backend = st or static(cells, design), backend
        L, coff = self.st["L"], self.st["coff"]
        cache = cells.setdefault("_nbr", {})
        ptr, idx, ends, base = [], [], [], 0
        for r in range(len(coff) - 1):
            n = int(coff[r + 1] - coff[r])
            if cache.get(r) is None:
                cache[r] = neighbours(L["xy"](r))
            # the kernels index adjacency without bounds checks: validate it as cpu.prepare does
            pr, ix, en, _, _ = count_inputs(cache[r], np.zeros(n, np.int8), np.ones(n, bool), L["K"])
            ptr.append(pr[:-1] + base)
            ends.append(en + base)
            idx.append(ix.astype(np.int32))
            base += len(ix)
        cat = lambda parts, dt, shape: np.concatenate(parts).astype(dt) if parts else np.zeros(shape, dt)  # noqa: E731
        host = [coff.astype(np.int64), np.asarray(cells["cell_type"]).astype(np.int32),
                np.asarray(cells["x"]).astype(np.float64), np.asarray(cells["y"]).astype(np.float64),
                cat(ptr, np.int64, 0), cat(ends, np.int64, (0, R)), cat(idx, np.int32, 0)]
        self.tail = [self.st["Mcum"], self.st["t_of"]]
        self.fixed = host
        if backend == "gpu":
            self.fixed, self.tail = [cuda.to_device(a) for a in host], [cuda.to_device(a) for a in self.tail]

    def evaluate(self, pl: dict, time, event, strata, keep_X: bool = False) -> dict:
        """One batch of planned studies -> z [S, J], maxT [S], stage seconds (and X [S, P', J] when keep_X)."""
        st, gpu = self.st, self.backend == "gpu"
        t0 = perf_counter()
        reg, K, T = pl["reg"], st["L"]["K"], len(st["L"]["focal"])
        S, P = pl["patients"].shape
        box = np.empty((len(reg), 4))
        box[:, 0::2], box[:, 1::2] = -np.inf, np.inf
        if st["fov"] is not None:  # random_crop and crop_mask's float64 expressions, elementwise
            xy0 = st["lo"][reg] + pl["a"] * st["w"][reg]
            crop = ~st["nocrop"][reg]
            box[crop, 0::2], box[crop, 1::2] = xy0[crop], xy0[crop] + float(st["fov"])
        ncell = np.diff(st["coff"])[reg]
        ooff = np.concatenate([[0], np.cumsum(ncell)]).astype(np.int64)
        rptr = np.concatenate([[0], np.cumsum(np.bincount(pl["row"], minlength=S * P))]).astype(np.int64)
        strata = np.asarray(strata)
        if strata.dtype.kind == "f" and not np.isfinite(strata).all():
            raise ValueError("strata must be finite")  # screen_studies drops such rows; this path has none
        codes = np.stack([np.unique(row, return_inverse=True)[1].ravel() for row in strata]).astype(np.int64)
        per_look = [reg, pl["hi"], pl["lo"], pl["ihi"], pl["ilo"], box, ooff]
        shapes = [((int(ooff[-1]),), np.int8), ((len(reg), K), np.float64), ((len(reg), R * K), np.int32),
                  ((len(reg), T * K * R * 2), np.float64), ((len(reg), T), np.int32)]
        stages = {"host_boxes_s": perf_counter() - t0}
        if not gpu:
            t0 = perf_counter()
            work = [np.empty(s, d) for s, d in shapes]
            X = np.empty((S, P, T * K * R * 2))
            _run_cpu(reg, *self.fixed, *per_look[1:], *self.tail, *work, rptr, X)
            stages["reimage_features_s"] = perf_counter() - t0
            from stats import screen_batch
            t0 = perf_counter()
            z = np.stack([screen_batch(X[s], np.asarray(time[s], float)[None], np.asarray(event[s], float)[None],
                                       strata=strata[s], backend="numpy")["z"][0] for s in range(S)])
            stages["score_s"] = perf_counter() - t0
        else:
            from stats.gpu_score import SMAX, _study_moments
            if codes.max() >= SMAX:
                raise ValueError(f"at most {SMAX} strata per study")

            def lap(name, t):
                cuda.synchronize()
                stages[name] = perf_counter() - t
                return perf_counter()
            t0 = perf_counter()
            dl = [cuda.to_device(a) for a in per_look]
            dscore = [cuda.to_device(np.ascontiguousarray(a, dtype=dt))
                      for a, dt in ((codes, np.int64), (time, np.float64), (event, np.float64))]
            work = [cuda.device_array(s, d) for s, d in shapes]
            X = cuda.device_array((S, P, T * K * R * 2))
            U, Ixx = cuda.device_array(X.shape[::2]), cuda.device_array(X.shape[::2])
            t0 = lap("upload_alloc_s", t0)
            _looks_kernel[(len(reg) + 63) // 64, 64](dl[0], *self.fixed[:3], *self.fixed[3:], *dl[1:], *self.tail, *work)
            t0 = lap("looks_kernel_s", t0)
            _patients_kernel[(S * P + 127) // 128, 128](cuda.to_device(rptr), work[3], work[4], X)
            t0 = lap("patients_kernel_s", t0)
            J = X.shape[2]
            _study_moments[((J + 31) // 32, (S + 3) // 4), (32, 4)](X, *dscore, U, Ixx)
            t0 = lap("score_kernel_s", t0)
            U, Ixx = U.copy_to_host(), Ixx.copy_to_host()
            good = Ixx > 0  # as stats.screen.screen_studies
            z = np.where(good, U / np.sqrt(np.where(good, Ixx, 1.0)), 0.0)
            stages["download_z_s"] = perf_counter() - t0
            free, total = cuda.current_context().get_memory_info() if not cuda.config.ENABLE_CUDASIM else (0, 0)
            stages["device_bytes_in_use"] = int(total - free)  # sampled with every buffer of the batch alive
            if keep_X:
                X = X.copy_to_host()
        out = {"z": z, "maxT": np.abs(z).max(1), "stages": stages, "looks": len(reg)}
        if keep_X:
            out["X"] = X
        return out
