"""Sufficit Phase 0: the error-algebra IR, minimal seed.

A Certified value = (value, err, tier, provenance, fail_p). err bounds the
distance to the true value (abs for scalars, 2-norm for vectors) and composes
through rewrites the way derivatives compose through autodiff. Tiers degrade
to the weakest input; provenance records which rewrites produced the bound;
fail_p is the probability the bound is wrong (0 for deterministic rewrites),
accumulating through composition by union bound.

Bounds are exact-arithmetic: floating-point rounding is not yet carried.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Callable, Tuple

import numpy as np


class Tier(IntEnum):
    EMPIRICAL = 0   # statistically validated
    ASYMPTOTIC = 1  # valid in a stated limit
    RIGOROUS = 2    # machine-checkable bound


@dataclass(frozen=True)
class Certified:
    value: Any                 # float or ndarray
    err: float                 # |value - true| (2-norm for vectors)
    tier: Tier
    provenance: Tuple[str, ...]
    fail_p: float = 0.0        # P(bound is wrong); composes by union bound

    def _combine(self, other: "Certified", value, err, note: str) -> "Certified":
        return Certified(value, err, min(self.tier, other.tier),
                         self.provenance + other.provenance + (note,),
                         min(1.0, self.fail_p + other.fail_p))

    def __add__(self, other: "Certified") -> "Certified":
        return self._combine(other, self.value + other.value,
                             self.err + other.err, "add")

    def __sub__(self, other: "Certified") -> "Certified":
        return self._combine(other, self.value - other.value,
                             self.err + other.err, "sub")

    def __mul__(self, other: "Certified") -> "Certified":
        # scalars only: |ab - a'b'| <= |a'|db + |b'|da + da*db
        err = (abs(self.value) * other.err + abs(other.value) * self.err
               + self.err * other.err)
        return self._combine(other, self.value * other.value, err, "mul")


def lipschitz(f: Callable, L: float, c: Certified, note: str) -> Certified:
    """Rewrite: push a certified value through an L-Lipschitz map."""
    return Certified(f(c.value), L * c.err, c.tier, c.provenance + (note,),
                     c.fail_p)


def lowrank_matvec(K: np.ndarray, q: np.ndarray, rank: int) -> Certified:
    """Rewrite: compress K@q to numerical rank. ||K - K_r||_2 = sigma_{r+1},
    so ||K_r q - K q||_2 <= sigma_{r+1} ||q||_2. Rigorous (exact arithmetic).
    """
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
    value = (U[:, :rank] * s[:rank]) @ (Vt[:rank] @ q)
    err = float(s[rank] * np.linalg.norm(q)) if rank < len(s) else 0.0
    return Certified(value, err, Tier.RIGOROUS,
                     (f"lowrank r={rank} sigma={err:.3g}",))


def lowrank_matvec_to_tol(K: np.ndarray, q: np.ndarray, tol: float) -> Certified:
    """Inverse rewrite: smallest SVD rank whose certified error meets tol."""
    # ponytail: SVD computed twice (here and in lowrank_matvec); fine at toy scale
    s = np.linalg.svd(K, compute_uv=False)
    tail = s * float(np.linalg.norm(q))
    hits = np.nonzero(tail <= tol)[0]
    rank = int(hits[0]) if len(hits) else len(s)
    return lowrank_matvec(K, q, rank)


def randomized_lowrank_matvec_to_tol(K: np.ndarray, q: np.ndarray, tol: float,
                                     n_probes: int = 10, rng=None) -> Certified:
    """Inverse rewrite: grow the sketch by rank doubling until the probe
    certifies err <= tol (adaptive range finder). fail_p union-bounds over
    every probe round the search consumed, not just the last one."""
    rng = np.random.default_rng(rng)
    max_rank = min(K.shape)
    rank, rounds = 1, 0
    while True:
        rounds += 1
        c = randomized_lowrank_matvec(K, q, rank, n_probes, rng)
        if c.err <= tol:
            break
        if rank >= max_rank:
            raise ValueError(f"cannot certify tol={tol:.3g}: full-rank probe "
                             f"residual is {c.err:.3g} (floating-point floor)")
        rank = min(2 * rank, max_rank)
    return replace(c, provenance=(f"rand-probe-adaptive r={rank} "
                                  f"rounds={rounds} k={n_probes}",),
                   fail_p=min(1.0, rounds * 10.0 ** (-n_probes)))


def randomized_lowrank_matvec(K: np.ndarray, q: np.ndarray, rank: int,
                              n_probes: int = 10, rng=None) -> Certified:
    """Rewrite: compress K@q via the randomized range finder, certified a
    posteriori by Gaussian probes (Halko-Martinsson-Tropp 2011, eq. 4.3):

        ||(I - QQ*)K||_2 <= 10 sqrt(2/pi) max_i ||(I - QQ*)K w_i||_2

    with failure probability 10^-n_probes over the probe draw. K is accessed
    only through matvec products, so this is the black-box compression
    detector: no SVD, no entrywise access assumed."""
    rng = np.random.default_rng(rng)
    m = K.shape[1]
    Q, _ = np.linalg.qr(K @ rng.standard_normal((m, rank)))
    value = Q @ (Q.T @ (K @ q))
    KW = K @ rng.standard_normal((m, n_probes))
    residuals = KW - Q @ (Q.T @ KW)
    norm_bound = 10.0 * math.sqrt(2.0 / math.pi) \
        * float(np.max(np.linalg.norm(residuals, axis=0)))
    return Certified(value, norm_bound * float(np.linalg.norm(q)),
                     Tier.RIGOROUS,
                     (f"rand-probe r={rank} k={n_probes}",),
                     fail_p=10.0 ** (-n_probes))


def _far_geometry(q, src, center, z):
    """(A, rho) for the multipole tail bound; rejects targets inside the
    source radius, where the bound is invalid."""
    r = float(np.max(np.abs(src - center)))
    d = abs(z - center)
    if d <= r:
        raise ValueError(f"target distance {d:.3g} <= source radius {r:.3g}: "
                         "multipole tail bound invalid")
    return float(np.sum(np.abs(q))), r / d


def multipole_far_potential(q: np.ndarray, src: np.ndarray, center: complex,
                            z: complex, p: int) -> Certified:
    """Rewrite: truncate the 2D multipole expansion of sum_j q_j log|z - src_j|
    at order p (Greengard-Rokhlin tail bound). Requires |z - center| strictly
    outside the source radius."""
    A, rho = _far_geometry(q, src, center, z)
    dz = src - center
    w = z - center
    total = float(np.sum(q)) * np.log(w)
    for k in range(1, p + 1):
        total += (-np.sum(q * dz**k) / k) / w**k
    return Certified(float(total.real), _tail(A, rho, p), Tier.RIGOROUS,
                     (f"multipole p={p} rho={rho:.3g}",))


def _tail(A: float, rho: float, p: int) -> float:
    """Greengard-Rokhlin tail bound at truncation order p."""
    return A / (p + 1) * rho ** (p + 1) / (1 - rho)


def _min_order(A: float, rho: float, tol: float) -> int:
    """Minimal p with _tail(A, rho, p) <= tol: closed-form sufficient order
    from the geometric part, then walked down."""
    if A == 0 or rho == 0:
        return 0
    t = tol * (1 - rho) / A
    p = max(0, math.ceil(math.log(t) / math.log(rho)) - 1) if t < 1 else 0
    while p > 0 and _tail(A, rho, p - 1) <= tol:
        p -= 1
    return p


def multipole_to_tol(q: np.ndarray, src: np.ndarray, center: complex,
                     z: complex, tol: float) -> Certified:
    """Inverse rewrite: minimal truncation order whose tail bound meets tol."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    A, rho = _far_geometry(q, src, center, z)
    return multipole_far_potential(q, src, center, z, _min_order(A, rho, tol))


# ---------------------------------------------------------------- Phase 1:
# hierarchical certified treecode. Dual quadtree traversal; well-separated
# cell pairs use per-cell multipole expansions (order chosen per pair from
# the target's error budget), everything else is direct. Hand-compiled
# schedule — the search that should discover it is future work.

_RHO_MAX = 0.5      # worst-case separation ratio for accepting a far pair
_P_CAP = 300        # order beyond this means the tolerance is absurd


class _Cell:
    __slots__ = ("center", "half", "idx", "radius", "children",
                 "coeffs", "Q", "A", "local")

    def __init__(self, pts, idx, center, half):
        self.center, self.half, self.idx = center, half, idx
        self.radius = float(np.max(np.abs(pts[idx] - center))) if len(idx) else 0.0
        self.children = []
        self.coeffs = np.zeros(0, complex)   # multipole b_k, grown on demand
        self.Q = self.A = None
        self.local = None                    # local (Taylor) coeffs, FMM only


def _build(pts, idx, center, half, leaf_size, depth=0):
    cell = _Cell(pts, idx, center, half)
    if len(idx) > leaf_size and depth < 30:
        h = half / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                c = center + complex(sx * h, sy * h)
                sub = idx[(np.sign(pts[idx].real - center.real + 1e-300) == sx)
                          & (np.sign(pts[idx].imag - center.imag + 1e-300) == sy)]
                if len(sub):
                    cell.children.append(_build(pts, sub, c, h, leaf_size,
                                                depth + 1))
    return cell


def _root(pts, leaf_size):
    lo, hi = complex(pts.real.min(), pts.imag.min()), \
             complex(pts.real.max(), pts.imag.max())
    half = max(hi.real - lo.real, hi.imag - lo.imag) / 2 + 1e-12
    return _build(pts, np.arange(len(pts)), (lo + hi) / 2, half, leaf_size)


def _ensure_coeffs(S, src, q, p, ops, via_m2m=False):
    if S.A is None:
        S.Q, S.A = float(np.sum(q[S.idx])), float(np.sum(np.abs(q[S.idx])))
    have = len(S.coeffs)
    if have >= p:
        return
    if not via_m2m or not S.children:      # P2M: directly from sources
        dz, qs = src[S.idx] - S.center, q[S.idx]
        new = [-np.sum(qs * dz**k) / k for k in range(have + 1, p + 1)]
        ops["p2m"] += len(S.idx) * (p - have)
    else:                                  # M2M: lower-triangular, exact
        for c in S.children:
            _ensure_coeffs(c, src, q, p, ops, True)
        deltas = [c.center - S.center for c in S.children]
        new = np.zeros(p - have, complex)
        row = np.ones(1)                   # C(l-1, k-1) for k = 1..l
        for l in range(1, p + 1):
            if l > 1:
                row = np.concatenate([[1.0], row[:-1] + row[1:], [1.0]])
            if l <= have:
                continue
            ks = np.arange(1, l + 1)
            for c, d in zip(S.children, deltas):
                new[l - have - 1] += -c.Q * d**l / l \
                    + np.sum(c.coeffs[:l] * row * d ** (l - ks))
                ops["m2m"] += l
    S.coeffs = np.concatenate([S.coeffs, np.asarray(new, complex)])


def _plan(T0, S0, symmetric):
    """Dual traversal into far pairs (T, S, rho_wc) and direct pairs (T, S).
    symmetric additionally requires the target cell to be small relative to
    the pair distance, which local expansions (FMM) need to converge."""
    far, direct, stack = [], [], [(T0, S0)]
    while stack:
        T, S = stack.pop()
        d = abs(T.center - S.center)
        dmin = d - T.radius
        ok = dmin > 0 and S.radius / dmin <= _RHO_MAX
        if ok and symmetric:
            ok = d - S.radius > 0 and T.radius / (d - S.radius) <= _RHO_MAX
        if ok:
            far.append((T, S, S.radius / dmin))
        elif not T.children and not S.children:
            direct.append((T, S))
        elif S.children and (not T.children or S.half >= T.half):
            stack += [(T, c) for c in S.children]
        else:
            stack += [(c, S) for c in T.children]
    return far, direct


def treecode_potential(tgt: np.ndarray, src: np.ndarray, q: np.ndarray,
                       eps: float, leaf_size: int = 48):
    """Phase 1 rewrite: sum_j q_j log|z_i - src_j| for every target, each
    certified to pointwise error <= eps. Returns (Certified, stats); the
    Certified err is the 2-norm of the per-target bounds, so it composes
    with the rest of the algebra. stats counts kernel-equivalent ops."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    T0, S0 = _root(tgt, leaf_size), _root(src, leaf_size)
    far, direct = _plan(T0, S0, symmetric=False)

    cnt = np.zeros(len(tgt), int)          # far interactions per target
    for T, _, _ in far:
        cnt[T.idx] += 1

    out, bound = np.zeros(len(tgt)), np.zeros(len(tgt))
    ops = {"p2m": 0, "far": 0, "direct": 0}
    for T, S, rho in far:
        if S.A is None:
            S.Q, S.A = float(np.sum(q[S.idx])), float(np.sum(np.abs(q[S.idx])))
        p = _min_order(S.A, rho, eps / cnt[T.idx].max())
        if p > _P_CAP:
            raise ValueError(f"eps={eps:g} needs order {p} at rho={rho:.3g}")
        _ensure_coeffs(S, src, q, p, ops)
        w = tgt[T.idx] - S.center
        acc, wk = S.Q * np.log(np.abs(w)), np.ones_like(w)
        for k in range(p):
            wk = wk / w
            acc = acc + (S.coeffs[k] * wk).real
        out[T.idx] += acc
        bound[T.idx] += _tail(S.A, rho, p)
        ops["far"] += len(T.idx) * (p + 1)
    for T, S in direct:
        out[T.idx] += np.log(np.abs(tgt[T.idx, None] - src[None, S.idx])) @ q[S.idx]
        ops["direct"] += len(T.idx) * len(S.idx)

    total = sum(ops.values())
    stats = {"ops": total, "dense_ops": len(tgt) * len(src),
             "speedup": len(tgt) * len(src) / total,
             "max_bound": float(bound.max()),
             "far_pairs": len(far), "direct_pairs": len(direct), **ops}
    cert = Certified(out, float(np.linalg.norm(bound)), Tier.RIGOROUS,
                     (f"treecode eps={eps:g} far={len(far)} "
                      f"direct={len(direct)}",))
    return cert, stats


# ------------------------------------------------------------- full FMM:
# M2L translates a source cell's multipole into a local (Taylor) expansion
# at the target cell; locals pass down the tree exactly (L2L is polynomial
# recentering, L2P is evaluation), so the only new error is the M2L
# truncation, bounded rigorously below. No M2M: multipoles still come from
# sources directly, keeping an O(N log N) term in the upward pass.


def _m2l_tail(absb, Q, D, beta, qloc, p):
    """Bound on |M_p(z) - Local_qloc(z)| for |z - c_T| <= beta*D: Taylor
    tails of the log term (geometric) and each 1/(z-c_S)^k term (binomial
    series majorized geometrically once its term ratio drops below 1)."""
    t = _tail(abs(Q), beta, qloc)
    for k in range(1, p + 1):
        denom = 1 - beta * (qloc + k + 1) / (qloc + 2)
        if denom <= 0:
            return math.inf
        t += (absb[k - 1] / D**k) * math.comb(qloc + k, k - 1) \
            * beta ** (qloc + 1) / denom
    return t


def _m2l_coeffs(S, z0, p, qloc):
    """Translate S's p-term multipole about c_S into qloc-term local coeffs
    about c_T, where z0 = c_S - c_T. Exact for the p-term expansion."""
    b = S.coeffs[:p]
    sgn = (-1.0) ** np.arange(1, p + 1)
    zk = z0 ** -np.arange(1, p + 1)
    out = np.zeros(qloc + 1, complex)
    out[0] = S.Q * np.log(-z0) + np.sum(b * sgn * zk)
    combs = np.ones(p)                      # C(l+k-1, k-1) built row by row
    for l in range(1, qloc + 1):
        combs *= (l + np.arange(1, p + 1) - 1) / l
        out[l] = -S.Q / (l * z0**l) + np.sum(b * sgn * combs * zk) / z0**l
    return out


def _shift_poly(c, delta):
    """Recenter sum_l c_l s^l with s = t + delta: exact, same degree."""
    out = np.zeros(len(c), complex)
    for m in range(len(c)):
        comb, dp, acc = 1.0, 1.0 + 0j, 0j
        for l in range(m, len(c)):
            acc += comb * dp * c[l]
            comb *= (l + 1) / (l + 1 - m)
            dp *= delta
        out[m] = acc
    return out


def _pad_add(a, b):
    if len(a) < len(b):
        a, b = b, a
    out = a.copy()
    out[:len(b)] += b
    return out


def fmm_potential(tgt: np.ndarray, src: np.ndarray, q: np.ndarray,
                  eps: float, leaf_size: int = 48):
    """Full-FMM rewrite (minus M2M): per-target certified potential as in
    treecode_potential, but far pairs feed local expansions via M2L when
    that is cheaper than evaluating the multipole at every target."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    T0, S0 = _root(tgt, leaf_size), _root(src, leaf_size)
    far, direct = _plan(T0, S0, symmetric=True)

    cnt = np.zeros(len(tgt), int)
    for T, _, _ in far:
        cnt[T.idx] += 1

    out, bound = np.zeros(len(tgt)), np.zeros(len(tgt))
    ops = {"p2m": 0, "m2m": 0, "m2l": 0, "l2l": 0, "l2p": 0,
           "far_eval": 0, "direct": 0}
    n_m2l = 0
    for T, S, rho in far:
        if S.A is None:
            S.Q, S.A = float(np.sum(q[S.idx])), float(np.sum(np.abs(q[S.idx])))
        eps_pair = eps / cnt[T.idx].max()
        z0 = S.center - T.center
        D, beta = abs(z0), T.radius / abs(z0)
        p = _min_order(S.A, rho, eps_pair / 2)
        _ensure_coeffs(S, src, q, p, ops, via_m2m=True)
        absb = np.abs(S.coeffs[:p])
        qloc = _min_order(S.A + abs(S.Q), beta, eps_pair / 2)
        while qloc <= _P_CAP and _m2l_tail(absb, S.Q, D, beta, qloc, p) \
                > eps_pair / 2:
            qloc += 1
        while qloc > 0 and _m2l_tail(absb, S.Q, D, beta, qloc - 1, p) \
                <= eps_pair / 2:
            qloc -= 1
        if qloc <= _P_CAP and (qloc + 1) * (p + 1) < len(T.idx) * (p + 1):
            T.local = _m2l_coeffs(S, z0, p, qloc) if T.local is None \
                else _pad_add(T.local, _m2l_coeffs(S, z0, p, qloc))
            bound[T.idx] += _tail(S.A, rho, p) \
                + _m2l_tail(absb, S.Q, D, beta, qloc, p)
            ops["m2l"] += (qloc + 1) * (p + 1)
            n_m2l += 1
        else:                               # multipole eval is cheaper here
            p = _min_order(S.A, rho, eps_pair)
            if p > _P_CAP:
                raise ValueError(f"eps={eps:g} needs order {p} at rho={rho:.3g}")
            _ensure_coeffs(S, src, q, p, ops, via_m2m=True)
            w = tgt[T.idx] - S.center
            acc, wk = S.Q * np.log(np.abs(w)), np.ones_like(w)
            for k in range(p):
                wk = wk / w
                acc = acc + (S.coeffs[k] * wk).real
            out[T.idx] += acc
            bound[T.idx] += _tail(S.A, rho, p)
            ops["far_eval"] += len(T.idx) * (p + 1)

    def down(T, inherited):
        total = inherited if T.local is None else _pad_add(T.local, inherited)
        if T.children:
            for c in T.children:
                if len(total):
                    ops["l2l"] += len(total) ** 2
                    down(c, _shift_poly(total, c.center - T.center))
                else:
                    down(c, total)   # locals may exist deeper down
        elif len(total):
            w = tgt[T.idx] - T.center
            acc = np.zeros_like(w)
            for coef in total[::-1]:
                acc = acc * w + coef
            out[T.idx] += acc.real
            ops["l2p"] += len(T.idx) * len(total)
    down(T0, np.zeros(0, complex))

    for T, S in direct:
        out[T.idx] += np.log(np.abs(tgt[T.idx, None] - src[None, S.idx])) @ q[S.idx]
        ops["direct"] += len(T.idx) * len(S.idx)

    total = sum(ops.values())
    stats = {"ops": total, "dense_ops": len(tgt) * len(src),
             "speedup": len(tgt) * len(src) / total,
             "max_bound": float(bound.max()),
             "m2l_pairs": n_m2l, "eval_pairs": len(far) - n_m2l,
             "direct_pairs": len(direct), **ops}
    cert = Certified(out, float(np.linalg.norm(bound)), Tier.RIGOROUS,
                     (f"fmm eps={eps:g} m2l={n_m2l} "
                      f"eval={len(far) - n_m2l} direct={len(direct)}",))
    return cert, stats


# ------------------------------------------------- black-box kernels:
# a certified H-matrix. Only kernel(tgt_pts, src_pts) -> block is assumed —
# no expansions, no smoothness proofs. Each admissible block is compressed
# by the randomized range finder and certified a posteriori by Gaussian
# probes (operator-norm bound, so it holds for EVERY later charge vector);
# fail_p union-bounds over all blocks and rounds. Build cost is ~N^2
# kernel evals — certification requires touching every block once — so
# the value is amortization: each apply is cheap and certified.


class BlackboxHMatrix:
    """Guarantee per apply(q): pointwise error <= eps * ||q||_2, with the
    plan's stated fail_p. Tier RIGOROUS (exact arithmetic)."""

    def __init__(self, kernel, tgt, src, eps, leaf_size=48, n_probes=10,
                 rng=None):
        if eps <= 0:
            raise ValueError("eps must be positive")
        rng = np.random.default_rng(rng)
        far, direct = _plan(_root(tgt, leaf_size), _root(src, leaf_size),
                            symmetric=False)
        cnt = np.zeros(len(tgt), int)
        for T, _, _ in far:
            cnt[T.idx] += 1

        self.n_tgt, self.n_src = len(tgt), len(src)
        self.far, self.near = [], []
        self.fail_p, evals, flops = 0.0, 0, 0
        scale = 10.0 * math.sqrt(2.0 / math.pi)
        for T, S, _ in far:
            K = kernel(tgt[T.idx], src[S.idx])
            evals += K.size
            tol = eps / cnt[T.idx].max()
            rank, full = 4, min(K.shape)
            while True:
                Q, _ = np.linalg.qr(K @ rng.standard_normal((K.shape[1], rank)))
                R = K @ rng.standard_normal((K.shape[1], n_probes))
                R -= Q @ (Q.T @ R)
                beta = scale * float(np.max(np.linalg.norm(R, axis=0)))
                self.fail_p += 10.0 ** (-n_probes)
                flops += K.size * (rank + n_probes)
                if beta <= tol or rank >= full:
                    break
                rank = min(2 * rank, full)
            if beta > tol:
                raise ValueError(f"block {K.shape} not compressible to "
                                 f"{tol:.3g}: probe bound {beta:.3g}")
            # doubling overshoots; trim rigorously: ||K - U'V'|| <= beta
            # + sigma_{r'+1} of the certified factor, no new probes needed
            U2, s2, Vt2 = np.linalg.svd(Q.T @ K, full_matrices=False)
            keep = int(np.searchsorted(-(beta + s2), -tol)) if len(s2) else 0
            keep = max(keep, 1)
            flops += K.size * rank
            if keep * (K.shape[0] + K.shape[1]) < K.size:
                self.far.append((T.idx, S.idx, (Q @ U2[:, :keep]) * s2[:keep],
                                 Vt2[:keep], beta + (s2[keep] if keep < len(s2)
                                                     else 0.0)))
            else:               # factors would cost more than the block does
                self.near.append((T.idx, S.idx, K))
        for T, S in direct:
            K = kernel(tgt[T.idx], src[S.idx])
            evals += K.size
            self.near.append((T.idx, S.idx, K))
        self.stats = {"kernel_evals": evals, "setup_flops": flops,
                      "far_blocks": len(self.far),
                      "near_blocks": len(self.near), "fail_p": self.fail_p}

    def apply(self, q):
        out, bound, flops = np.zeros(self.n_tgt), np.zeros(self.n_tgt), 0
        for ti, si, U, V, beta in self.far:
            out[ti] += U @ (V @ q[si])
            bound[ti] += beta * float(np.linalg.norm(q[si]))
            flops += V.shape[0] * (len(ti) + len(si))
        for ti, si, K in self.near:
            out[ti] += K @ q[si]
            flops += len(ti) * len(si)
        stats = {"apply_flops": flops,
                 "dense_flops": self.n_tgt * self.n_src,
                 "speedup": self.n_tgt * self.n_src / flops,
                 "max_bound": float(bound.max())}
        cert = Certified(out, float(np.linalg.norm(bound)), Tier.RIGOROUS,
                         (f"blackbox-hmatrix far={len(self.far)} "
                          f"near={len(self.near)}",), self.fail_p)
        return cert, stats


# ------------------------------------------------------------- Phase 2:
# Gibbs-state observables via certified high-temperature cluster expansion.
# 2D Ising, square lattice, weight exp(betaJ * sum s_i s_j). The tanh
# expansion maps Z onto a polymer gas of connected even subgraphs with
# weight t^|edges|, t = tanh(betaJ), compatibility = vertex-disjointness.
# The Kotecky-Preiss condition (per-vertex form via a ghost polymer, cf.
# Friedli-Velenik Thm 5.4) with the rigorous walk-encoding count
# (# connected n-edge subgraphs through a vertex <= Delta^2n, Delta = 4)
# gives: for x_s = t e^(1+s) * 16 with x_s^4/(1-x_s) <= 1, the cluster
# terms of size >= L sum below e^(-s L) per site. This is the first
# rewrite with a VALIDITY REGION: outside it, refuse.
# Order cap L = 8: below 8 edges the only clusters are single polymers
# that are simple cycles (pairs enter at 4+4, non-cycle polymers at 8),
# so no Ursell machinery is needed yet. Raising the cap requires union
# polymers and pair clusters. The Delta^2n count is ~26x conservative in
# beta (certified radius betaJ < ~0.0167 vs true ~0.44); tightening the
# subgraph count is the named improvement path.


def _ising2d_anchored_cycles(max_edges):
    """Edge counts of simple cycles on Z^2 whose lexicographically minimal
    vertex is the origin, each counted once, up to max_edges edges."""
    found = []

    def extend(path, seen):
        x, y = path[-1]
        for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if nxt == (0, 0) and len(path) >= 3:
                if path[1] < path[-1]:      # one orientation per cycle
                    found.append(len(path))
                continue
            if nxt in seen or nxt < (0, 0) or len(path) >= max_edges:
                continue
            extend(path + [nxt], seen | {nxt})

    extend([(0, 0)], {(0, 0)})
    return found


def ising2d_logZ_density(beta: float, J: float = 1.0,
                         tol: float = None) -> Certified:
    """Phase 2 rewrite: log Z per site for the 2D Ising model at inverse
    temperature beta, certified by the cluster-expansion tail. Valid for
    every m x m torus with m >= 8 and for the thermodynamic limit.
    Raises outside the certified high-temperature region, or when the
    requested tol is unreachable at the order cap."""
    t = math.tanh(beta * J)
    if t == 0.0:
        return Certified(math.log(2.0), 0.0, Tier.RIGOROUS,
                         ("ising2d-cluster t=0 exact",))
    lo, hi = 0.0, 1.0                       # y* solves y^4 + y = 1
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid**4 + mid < 1 else (lo, mid)
    t_max = lo / (math.e * 16)
    if abs(t) >= t_max:
        raise ValueError(
            f"|tanh(beta J)|={abs(t):.4g} >= {t_max:.4g}: outside the "
            "certified high-temperature region (KP with Delta^2n counting)")
    L = 8
    s = math.log(lo / (abs(t) * math.e * 16))
    err = math.exp(-L * s)
    if tol is not None and err > tol:
        raise ValueError(f"certified tail {err:.3g} exceeds tol={tol:.3g} "
                         "at this temperature (order cap L=8)")
    f = math.log(2.0) + 2.0 * math.log(math.cosh(beta * J)) \
        + sum(t**n for n in _ising2d_anchored_cycles(L - 1))
    return Certified(f, err, Tier.RIGOROUS,
                     (f"ising2d-cluster L={L} s={s:.3g} t={t:.4g} "
                      "[Kotecky-Preiss]",))
