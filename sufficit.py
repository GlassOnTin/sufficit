"""Sufficit Phase 0: the error-algebra IR, minimal seed.

A Certified value = (value, err, tier, provenance, fail_p). err bounds the
distance to the true value (abs for scalars, 2-norm for vectors) and composes
through rewrites the way derivatives compose through autodiff. Tiers degrade
to the weakest input; provenance records which rewrites produced the bound;
fail_p is the probability the bound is wrong (0 for deterministic rewrites),
accumulating through composition by union bound.

The numpy-based rewrites carry exact-arithmetic bounds (declared in their
docstrings); the scalar Phase 2 pipelines carry floating-point rounding
too, via the directed-rounding Interval type ("+fp" in provenance).
"""
from __future__ import annotations

import functools
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


def _dn(x):
    return math.nextafter(x, -math.inf)


def _up(x):
    return math.nextafter(x, math.inf)


class Interval:
    """Directed-rounding interval via outward nextafter widening: field ops
    widen 1 ulp (IEEE round-to-nearest is within 1/2 ulp), transcendentals
    2 ulps under the ASSUMPTION of faithful (<= 1 ulp) libm — the one
    unverified assumption in this class. Used to carry FP error through
    the scalar Phase 2 pipelines; the numpy rewrites remain
    exact-arithmetic as declared in their docstrings."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if isinstance(lo, Interval):
            self.lo, self.hi = lo.lo, lo.hi
        else:
            self.lo = float(lo)
            self.hi = self.lo if hi is None else float(hi)

    def __add__(self, o):
        o = Interval(o)
        return Interval(_dn(self.lo + o.lo), _up(self.hi + o.hi))
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-Interval(o))

    def __rsub__(self, o):
        return Interval(o) + (-self)

    def __mul__(self, o):
        o = Interval(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(_dn(min(p)), _up(max(p)))
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Interval(o)
        if o.lo <= 0.0 <= o.hi:
            raise ZeroDivisionError("interval denominator spans 0")
        p = (self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi)
        return Interval(_dn(min(p)), _up(max(p)))

    def __rtruediv__(self, o):
        return Interval(o) / self

    def __pow__(self, n):                  # non-negative int; sign-safe
        r = Interval(1.0)
        for _ in range(n):
            r = r * self
        return r

    def _mono(self, f):                    # monotone libm call, 2-ulp widen
        return Interval(_dn(_dn(f(self.lo))), _up(_up(f(self.hi))))

    def exp(self):
        return self._mono(math.exp)

    def tanh(self):
        return self._mono(math.tanh)

    def log(self):
        if self.lo <= 0:
            raise ValueError("log of interval touching 0")
        return self._mono(math.log)

    def cosh(self):                        # even, minimum 1 at 0
        hi = _up(_up(math.cosh(max(abs(self.lo), abs(self.hi)))))
        lo = 1.0 if self.lo <= 0.0 <= self.hi \
            else _dn(_dn(math.cosh(min(abs(self.lo), abs(self.hi)))))
        return Interval(lo, hi)

    def abs(self):
        if self.lo <= 0.0 <= self.hi:
            return Interval(0.0, max(-self.lo, self.hi))
        return Interval(self.lo, self.hi) if self.lo > 0 else -self

    @property
    def mid(self):
        return (self.lo + self.hi) / 2

    @property
    def rad(self):                         # bound on |mid - true|
        m = self.mid
        return max(_up(m - self.lo), _up(self.hi - m))


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


def _compress_certified(K, tol, n_probes, scale, rng):
    """Adaptive certified low-rank factorization: (U, V, beta, rounds,
    flops) with ||K - UV||_2 <= beta <= tol (each probe round fails with
    prob 10^-n_probes), or None if full rank cannot certify tol."""
    rank, full, flops, rounds = 4, min(K.shape), 0, 0
    while True:
        rounds += 1
        Q, _ = np.linalg.qr(K @ rng.standard_normal((K.shape[1], rank)))
        R = K @ rng.standard_normal((K.shape[1], n_probes))
        R -= Q @ (Q.conj().T @ R)
        beta = scale * float(np.max(np.linalg.norm(R, axis=0)))
        flops += K.size * (rank + n_probes)
        if beta <= tol or rank >= full:
            break
        rank = min(2 * rank, full)
    if beta > tol:
        return None
    # doubling overshoots; trim rigorously: ||K - U'V'|| <= beta
    # + sigma_{r'+1} of the certified factor, no new probes needed
    U2, s2, Vt2 = np.linalg.svd(Q.conj().T @ K, full_matrices=False)
    keep = max(1, int(np.searchsorted(-(beta + s2), -tol)) if len(s2) else 0)
    flops += K.size * rank
    return ((Q @ U2[:, :keep]) * s2[:keep], Vt2[:keep],
            beta + (s2[keep] if keep < len(s2) else 0.0), rounds, flops)


def _butterfly_candidate(K, tpts, spts, tol, n_probes, scale, rng):
    """Try the butterfly rewrite on one block: deepest feasible ladder
    (capped at 3 levels), per-factor tolerance scaled from tol by a
    one-probe norm estimate. Admissible only if its a posteriori beta
    meets tol — a wrong guess just loses the competition."""
    m, n = K.shape
    L = min(3, int(math.log(max(min(m, n) // 8, 1), 4)))
    if L < 2 or min(m, n) < 512:
        return None
    w = rng.standard_normal(n)
    normest = float(np.linalg.norm(K @ w) / np.linalg.norm(w))
    # the probe estimator overshoots flat residuals by ~ scale*sqrt(m)
    # (it tracks the Frobenius norm); calibrate the per-factor tolerance
    # for it so the a posteriori beta lands near tol
    eps_rel = min(0.1, tol / (normest * scale * math.sqrt(m)))
    bf = ButterflyBlock(K, tpts, spts, L, eps=eps_rel,
                        n_probes=n_probes, rng=rng)
    return bf if bf.beta <= tol else None


class BlackboxHMatrix:
    """Guarantee per apply(q): pointwise error <= eps * ||q||_2, with the
    plan's stated fail_p. Tier RIGOROUS (exact arithmetic).

    NOTE (negative result, kept so it is not re-learned): demodulating a
    block by pair-direction phases D_T K D_S is a UNITARY transformation —
    singular values are identical, so per-block directional demodulation
    cannot reduce SVD ranks (measured: 12=12, 14=14 at k=150). The
    high-frequency island needs the genuine multi-level butterfly
    (row-split/column-merge with transfer operators), not implemented."""

    def __init__(self, kernel, tgt, src, eps, leaf_size=48, n_probes=10,
                 rng=None, try_butterfly=True):
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
        self.dtype, scale = np.float64, None
        n_lr = n_bf = n_dense = 0
        for T, S, _ in far:
            K = kernel(tgt[T.idx], src[S.idx])
            evals += K.size
            if scale is None:
                # complex kernels: real probes certify the real-restricted
                # norm; ||B||_C <= sqrt(2) * that, so the estimator pays it
                self.dtype = np.result_type(K)
                scale = 10.0 * math.sqrt(2.0 / math.pi) \
                    * (math.sqrt(2.0) if np.issubdtype(
                        self.dtype, np.complexfloating) else 1.0)
            tol = eps / cnt[T.idx].max()
            cands = []          # (stored_cost, op, beta)
            res = _compress_certified(K, tol, n_probes, scale, rng)
            if res is not None:
                U, V, beta, rounds, fl = res
                self.fail_p += rounds * 10.0 ** (-n_probes)
                flops += fl
                cands.append((V.shape[0] * (K.shape[0] + K.shape[1]),
                              (U, V), beta))
            if try_butterfly \
                    and np.issubdtype(self.dtype, np.complexfloating):
                bf = _butterfly_candidate(K, tgt[T.idx], src[S.idx], tol,
                                          n_probes, scale, rng)
                if bf is not None:
                    self.fail_p += 10.0 ** (-n_probes)
                    cands.append((bf.apply_flops, bf, bf.beta))
            if not cands:
                raise ValueError(f"block {K.shape} not compressible to "
                                 f"{tol:.3g}: full-rank probe failed")
            cost, op, beta = min(cands, key=lambda c: c[0])
            if cost < K.size:
                self.far.append((T.idx, S.idx, op, beta))
                n_bf += isinstance(op, ButterflyBlock)
                n_lr += not isinstance(op, ButterflyBlock)
            else:               # factors would cost more than the block does
                self.near.append((T.idx, S.idx, K))
                n_dense += 1
        for T, S in direct:
            K = kernel(tgt[T.idx], src[S.idx])
            evals += K.size
            self.near.append((T.idx, S.idx, K))
        self.stats = {"kernel_evals": evals, "setup_flops": flops,
                      "far_blocks": len(self.far),
                      "near_blocks": len(self.near), "lr_blocks": n_lr,
                      "butterfly_blocks": n_bf, "dense_blocks": n_dense,
                      "fail_p": self.fail_p}

    def apply(self, q):
        out = np.zeros(self.n_tgt, self.dtype)
        bound, flops = np.zeros(self.n_tgt), 0
        for ti, si, op, beta in self.far:
            if isinstance(op, ButterflyBlock):
                out[ti] += op._raw(q[si])
                flops += op.apply_flops
            else:
                U, V = op
                out[ti] += U @ (V @ q[si])
                flops += V.shape[0] * (len(ti) + len(si))
            bound[ti] += beta * float(np.linalg.norm(q[si]))
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
# Friedli-Velenik Thm 5.4) with the Eulerian-circuit count (even connected
# subgraphs are Eulerian: each of the n edges traversed once, never
# departing along a used edge, so <= 4*3^(n-1) through a vertex) gives:
# the cluster terms of size >= L sum below e^(-s L) per site, s from
# _kp_rate. This is the first rewrite with a VALIDITY REGION: outside
# it, refuse.
# Order cap L = 12: single polymers (cycles and two-cycle unions) and
# Ursell pair clusters cover every cluster below 12 edges; triples of
# polymers enter at 4+4+4 = 12, so raising further needs phi^T for
# n >= 3. The certified radius is betaJ < ~0.085 vs
# the true critical ~0.4407 (~5x conservative; the remaining gap is the
# 3^n walk constant vs the true even-subgraph growth ~2.64^n plus the
# factor e that KP itself pays).


_E_IV = Interval(_dn(math.e), _up(math.e))


def _kp_rate(t_abs: Interval) -> Interval:
    """Tilt rate s of the KP tail e^(-s L) at activity |t|; raises outside
    the certified region. Counting: connected even subgraphs are Eulerian,
    so a circuit from v traverses each of its n edges exactly once and
    never departs along a used edge — at most Delta*(Delta-1)^(n-1)
    = 4*3^(n-1) walks. Per-vertex KP condition: (4/3) u^4/(1-u) <= 1 with
    u = 3|t| e^(1+s); u* is the root of 4u^4 + 3u = 3, lower-bounded
    rigorously by interval bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 4 * Interval(mid)**4 + 3 * Interval(mid) - 3
        if p.hi < 0:
            lo = mid
        elif p.lo > 0:
            hi = mid
        else:
            break                       # ambiguous at FP resolution; lo holds
    t_max = Interval(lo) / (3 * _E_IV)
    if t_abs.hi >= t_max.lo:
        raise ValueError(
            f"|tanh(beta J)|={t_abs.hi:.4g} >= {t_max.lo:.4g}: outside the "
            "certified high-temperature region (KP, Eulerian counting)")
    return (Interval(lo) / (3 * _E_IV * t_abs)).log()


def _ising2d_cycles(max_edges):
    """Simple cycles on Z^2 with lexicographically minimal vertex at the
    origin (canonical translation-class representatives), as
    (edge-frozenset, vertex-frozenset) pairs, up to max_edges edges."""
    found = []

    def extend(path, seen):
        x, y = path[-1]
        for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if nxt == (0, 0) and len(path) >= 3:
                if path[1] < path[-1]:      # one orientation per cycle
                    edges = frozenset(tuple(sorted(p)) for p in
                                      zip(path, path[1:] + [(0, 0)]))
                    found.append((edges, frozenset(path)))
                continue
            if nxt in seen or nxt < (0, 0) or len(path) >= max_edges:
                continue
            extend(path + [nxt], seen | {nxt})

    extend([(0, 0)], {(0, 0)})
    return found


def _ising2d_anchored_cycles(max_edges):
    """Edge counts of the canonical cycles (kept for tests/inspection)."""
    return [len(e) for e, _ in _ising2d_cycles(max_edges)]


def _shift(pts, d):
    return frozenset((p[0] + d[0], p[1] + d[1]) for p in pts)


def _diffs(V1, V2):
    """Offsets d with (V2 + d) meeting V1."""
    return {(a[0] - b[0], a[1] - b[1]) for a in V1 for b in V2}


@functools.lru_cache(maxsize=None)
def _ising2d_polymer_shapes(max_edges=10):
    """Connected even subgraphs of Z^2 up to max_edges edges, canonical
    (min vertex at origin): simple cycles plus edge-disjoint,
    vertex-sharing unions of two cycles. Triples of cycles enter at 12
    edges, above the cap."""
    cycles = _ising2d_cycles(max_edges)
    shapes = dict(cycles)
    for e1, v1 in cycles:
        for e2, v2 in cycles:
            if len(e1) + len(e2) > max_edges:
                continue
            for d in _diffs(v1, v2):
                e2t = frozenset((( a[0] + d[0], a[1] + d[1]),
                                 (b[0] + d[0], b[1] + d[1]))
                                for a, b in e2)
                if e1 & e2t:
                    continue            # shares an edge: not a valid union
                verts = v1 | _shift(v2, d)
                mv = min(verts)
                off = (-mv[0], -mv[1])
                E = frozenset(((a[0] + off[0], a[1] + off[1]),
                               (b[0] + off[0], b[1] + off[1]))
                              for a, b in (e1 | e2t))
                shapes[E] = _shift(verts, off)
    return tuple(shapes.items())


@functools.lru_cache(maxsize=None)
def _ising2d_logz_coeffs():
    """Per-site coefficients of the high-temperature series in
    t = tanh(beta J), exact through t^10: anchored single polymers plus
    Ursell pair clusters, -(1/2) over ordered incompatible pairs
    (phi^T = -1). Reproduces the known series
    log Z/N = log 2 + 2 log cosh + t^4 + 2 t^6 + (9/2) t^8 + 12 t^10."""
    shapes = _ising2d_polymer_shapes(10)
    coef = {}
    for E, V in shapes:
        coef[len(E)] = coef.get(len(E), 0.0) + 1.0
    small = [(E, V) for E, V in shapes if len(E) <= 6]
    for E1, V1 in small:
        for E2, V2 in small:
            n = len(E1) + len(E2)
            if n < 12:
                coef[n] = coef.get(n, 0.0) - 0.5 * len(_diffs(V1, V2))
    return coef


def _ising2d_psi_coeffs(S):
    """Coefficients of Psi(S) — the cluster sum touching the vertex set S
    — exact through total size 10: singles, plus Ursell pairs whose
    union touches S via 2*(touching x incompatible-anywhere) minus
    (both touching), all reduced to integer offset counting."""
    shapes = _ising2d_polymer_shapes(10)
    Sf = frozenset(S)
    coef = {}
    for E, V in shapes:
        c = len(_diffs(Sf, V))
        if c:
            coef[len(E)] = coef.get(len(E), 0.0) + c
    small = [(E, V) for E, V in shapes if len(E) <= 6]
    touch = [[_shift(V, d) for d in _diffs(Sf, V)] for _, V in small]
    for i, (E1, V1) in enumerate(small):
        for j, (E2, V2) in enumerate(small):
            n = len(E1) + len(E2)
            if n < 12:
                k = len(_diffs(V1, V2))
                both = sum(1 for a in touch[i] for b in touch[j] if a & b)
                coef[n] = coef.get(n, 0.0) - len(touch[i]) * k + 0.5 * both
    return coef


def ising2d_logZ_density(beta: float, J: float = 1.0,
                         tol: float = None) -> Certified:
    """Phase 2 rewrite: log Z per site for the 2D Ising model at inverse
    temperature beta, certified by the cluster-expansion tail. Valid for
    every m x m torus with m >= 12 (wrapping clusters below 12
    edges would otherwise escape both truncation and tail) and for the
    thermodynamic limit.
    Raises outside the certified high-temperature region, or when the
    requested tol is unreachable at the order cap."""
    bJ = Interval(beta) * J
    if beta * J == 0.0:
        v = Interval(2.0).log()
        return Certified(v.mid, v.rad, Tier.RIGOROUS,
                         ("ising2d-cluster t=0 +fp",))
    t = bJ.tanh()
    L = 12
    s = _kp_rate(t.abs())
    tail = (Interval(-float(L)) * s).exp()
    f = Interval(2.0).log() + 2 * bJ.cosh().log()
    for n, cf in sorted(_ising2d_logz_coeffs().items()):
        f = f + cf * t**n
    err = _up(tail.hi + f.rad)
    if tol is not None and err > tol:
        raise ValueError(f"certified error {err:.3g} exceeds tol={tol:.3g} "
                         "at this temperature (order cap L=8)")
    return Certified(f.mid, err, Tier.RIGOROUS,
                     (f"ising2d-cluster L={L} s={s.lo:.3g} t={t.mid:.4g} "
                      "+fp [Kotecky-Preiss]",))


def _connected_pinned_subgraphs(a, b, max_edges):
    """Connected edge-subgraphs of Z^2 containing vertex a, <= max_edges
    edges, with odd-degree set exactly {a, b}. Ban-list recursion generates
    each subgraph once (cross-validated against brute force in the tests)."""
    def about(v):
        x, y = v
        return [tuple(sorted([v, u])) for u in
                ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))]

    results = []

    def rec(C, V, deg, banned):
        if C and {v for v, d in deg.items() if d % 2} == {a, b}:
            results.append(frozenset(C))
        if len(C) == max_edges:
            return
        cand = sorted({e for v in (V or [a]) for e in about(v)
                       if e not in C and e not in banned})
        banned = set(banned)
        for e in cand:
            deg2 = dict(deg)
            for v in e:
                deg2[v] = deg2.get(v, 0) + 1
            rec(C | {e}, V | set(e), deg2, banned)
            banned.add(e)

    rec(frozenset(), set(), {}, set())
    return results


def ising2d_bond_correlation(beta: float, J: float = 1.0,
                             tol: float = None) -> Certified:
    """Phase 2 rewrite: <s_a s_b> for a nearest-neighbor pair of the 2D
    Ising model, via pinned clusters: every subgraph with odd set {a,b}
    is one connected pinned polymer w0 times an even gas off its vertices,
    so <s_a s_b> = sum_w0 t^|w0| exp(-Psi(V(w0))) with Psi the cluster sum
    touching V(w0). Errors: pinned tail (pinned polymers have exactly two
    odd vertices, hence an Eulerian path a->b: count <= 4*3^(n-1);
    dressing bounded by e^((n+1)B)), Psi truncation at size 8 (tilted KP,
    per vertex), propagated through exp. Same validity region and
    torus/limit scope (m >= 12) as ising2d_logZ_density."""
    if beta * J == 0.0:                 # no computation: exactly 0
        return Certified(0.0, 0.0, Tier.RIGOROUS, ("ising2d-pinned t=0",))
    t = (Interval(beta) * J).tanh()
    ta = t.abs()
    L1, L2 = 8, 12          # pinned-sum order; cluster-dressing order
    s = _kp_rate(ta)
    u0 = 3 * _E_IV * ta
    B = (Interval(4.0) / 3) * u0**4 / (1 - u0)   # per-vertex cluster bound
    eB = B.exp()
    y = 3 * ta * eB
    if y.hi >= 1.0:
        raise ValueError("too close to the region boundary to certify")
    total, e2 = Interval(0.0), Interval(0.0)
    dfac = (Interval(-float(L2)) * s).exp()
    for C in _connected_pinned_subgraphs((0, 0), (1, 0), L1 - 1):
        S = {v for e in C for v in e}
        psi = Interval(0.0)
        for n, cf in sorted(_ising2d_psi_coeffs(frozenset(S)).items()):
            psi = psi + cf * t**n
        damp = (-psi).exp()
        total = total + t ** len(C) * damp
        e2 = e2 + ta ** len(C) * damp * ((len(S) * dfac).exp() - 1)
    e1 = (Interval(4.0) / 3) * eB * y**L1 / (1 - y)
    err = _up(e1.hi + e2.hi + total.rad)
    if tol is not None and err > tol:
        raise ValueError(f"certified error {err:.3g} exceeds tol={tol:.3g} "
                         "at this temperature (order cap L=8)")
    return Certified(total.mid, err, Tier.RIGOROUS,
                     (f"ising2d-pinned L={L1}/{L2} s={s.lo:.3g} t={t.mid:.4g} "
                      "+fp [Kotecky-Preiss]",))


# ------------------------------------------------------------- Phase 3:
# smeared spectral functions (Hansen-Lupo-Tantalo mold). The resolution
# sigma is part of the query. Data: C(t) = int_0^inf e^(-wt) rho(w) dw,
# t = 1..N, with rho >= 0 (declared physical assumption). Coefficients
# g_t (t >= 2) reconstruct the Gaussian smearing kernel; validity never
# depends on how g was found, because the certificate is a posteriori:
#   |sum_t g_t e^(-wt) - Delta(w)| <= c * e^(-w)   for all w >= 0
# with c a rigorous sup bound (grid + per-cell Lipschitz + analytic
# tail), whence by positivity   |value - true| <= c * C(1).
# Certificate is exact-arithmetic (FP not carried in this pipeline).


def _gauss_laplace(tau, omega, sigma):
    """int_0^inf e^(-tau w) N(w; omega, sigma) dw, overflow-safe."""
    from scipy.special import erfcx
    y = (sigma * sigma * tau - omega) / (sigma * math.sqrt(2))
    if y < 0:
        return 0.5 * math.exp(0.5 * sigma**2 * tau**2 - omega * tau) \
            * math.erfc(y)
    return 0.5 * float(erfcx(y)) * math.exp(-omega * omega / (2 * sigma**2))


@functools.lru_cache(maxsize=None)
def _hlt_solve(N, omega, sigma, ridge=None):
    """Solve for g_t (t = 2..N) minimizing the e^(2w)-weighted L2 kernel
    deviation, then certify c = sup_w |deviation| e^w rigorously. With no
    ridge given, scans the two ridges that win in practice and keeps the
    best-certifying g — the bound is a posteriori, so the scan cannot
    compromise validity. Cached: callers treat g as read-only."""
    if ridge is None:
        best = min((_hlt_solve(N, omega, sigma, r)
                    for r in (1e-8, 1e-10)), key=lambda gc: gc[1])
        return best
    ts = np.arange(2, N + 1)
    A = 1.0 / (ts[:, None] + ts[None, :] - 2.0)
    f = np.array([_gauss_laplace(t - 2.0, omega, sigma) for t in ts])
    g = np.linalg.solve(A + ridge * np.eye(len(ts)), f)

    wmax = omega + max(8 * sigma, sigma * sigma) + 4.0
    w = np.concatenate([np.linspace(0, 1, 1_000_001),
                        np.linspace(1, wmax, 1_000_001)[1:]])
    delta = np.exp(-(w - omega) ** 2 / (2 * sigma**2)) \
        / (sigma * math.sqrt(2 * math.pi))
    h = -delta * np.exp(w)
    for gt, t in zip(g, ts):
        h += gt * np.exp(-w * (t - 1.0))
    ah = np.abs(h)
    # per-cell Lipschitz bound on h: the g-part decays (evaluate at the
    # cell's left edge); the Gaussian part uses |Delta'| = |w-omega|/s^2
    # * Delta with cellwise maxima
    step = np.diff(w)
    gpart = np.zeros(len(w))
    for gt, t in zip(g, ts):
        gpart += abs(gt) * (t - 1.0) * np.exp(-w * (t - 1.0))
    dmax = np.maximum(delta[:-1], delta[1:])
    inside = (w[:-1] <= omega) & (w[1:] >= omega)
    dmax[inside] = 1.0 / (sigma * math.sqrt(2 * math.pi))
    dist = np.maximum(np.abs(w[:-1] - omega), np.abs(w[1:] - omega))
    lip = gpart[:-1] + (dist / sigma**2 + 1.0) * dmax * np.exp(w[1:])
    c_grid = float(np.max(np.maximum(ah[:-1], ah[1:]) + lip * step / 2))
    # tail w >= wmax: g-part decays from wmax; Delta e^w is decreasing
    # there (wmax > omega + sigma^2)
    c_tail = sum(abs(gt) * math.exp(-wmax * (t - 1.0))
                 for gt, t in zip(g, ts)) \
        + math.exp(-(wmax - omega) ** 2 / (2 * sigma**2) + wmax) \
        / (sigma * math.sqrt(2 * math.pi))
    return g, max(c_grid, c_tail)


def smeared_spectral(C: np.ndarray, omega: float, sigma: float,
                     cov: np.ndarray = None, z: float = 5.0) -> Certified:
    """Phase 3 rewrite: Gaussian-smeared spectral value
    int N(w; omega, sigma) rho(w) dw from Euclidean correlator data
    C[t-1] = C(t), t = 1..N. ASSUMES rho >= 0 (declared; physical
    spectral densities are). Exact data: RIGOROUS with err = c*C(1).
    With covariance cov: adds z-sigma statistical error on the value and
    on C(1); tier degrades to EMPIRICAL (Gaussian-noise assumption) with
    fail_p = 2*erfc(z/sqrt(2))."""
    N = len(C)
    g, c = _hlt_solve(N, omega, sigma)
    value = float(g @ C[1:])
    if cov is None:
        return Certified(value, c * float(C[0]), Tier.RIGOROUS,
                         (f"hlt-smeared omega={omega:g} sigma={sigma:g} "
                          f"c={c:.3g} assumes rho>=0",))
    stat = z * math.sqrt(float(g @ cov[1:, 1:] @ g))
    err = c * (float(C[0]) + z * math.sqrt(float(cov[0, 0]))) + stat
    return Certified(value, err, Tier.EMPIRICAL,
                     (f"hlt-smeared omega={omega:g} sigma={sigma:g} "
                      f"c={c:.3g} z={z:g} assumes rho>=0",),
                     fail_p=2 * math.erfc(z / math.sqrt(2)))


# ------------------------------------------------------------- Phase 4:
# Mori-Zwanzig closures. Rigorous tier: linear slow-fast systems, where
# the memory kernel K(s) = A12 e^(A22 s) A21 decays at the fast sector's
# spectral gap and the Markovian closure carries a Gronwall bound with
# computable constants (log-norms, block norms) — the bound's provenance
# is the gap, and without one the rewrite refuses. Empirical tier:
# distribution-free conformal calibration for closures of ANY system
# (nonlinear included), with rigorous failure probability 1/(n_cal+1)
# under exchangeability of initial conditions — the honest certificate
# this phase exists to ship. Both exact-arithmetic.


def _lognorm(M):
    """2-norm logarithmic norm: ||e^(Mt)|| <= e^(lognorm(M) t)."""
    return float(np.linalg.eigvalsh((M + M.T) / 2)[-1])


def mz_closure_linear(A: np.ndarray, k: int, x0: np.ndarray,
                      T: float) -> Certified:
    """Phase 4 rewrite, rigorous tier: x_slow(T) for dx/dt = Ax with the
    first k coordinates slow observables (x0 is the FULL initial state),
    via the Markovian closure Ar = A11 - A12 A22^{-1} A21. Error bound:
    ||K(s)|| <= kappa e^(-mu s) with mu the fast-sector gap (-lognorm of
    A22), Gronwall through the reduced propagator; a nonzero fast
    initial condition contributes its decaying transient
    kappa12 ||x20|| G2. Refuses when mu <= 0."""
    from scipy.linalg import expm
    A = np.asarray(A, float)
    A11, A12 = A[:k, :k], A[:k, k:]
    A21, A22 = A[k:, :k], A[k:, k:]
    mu = -_lognorm(A22)
    if mu <= 0:
        raise ValueError(f"fast-sector log-norm {-mu:.3g} >= 0: no spectral "
                         "gap, memory kernel not certifiably decaying")
    k12 = float(np.linalg.norm(A12, 2))
    kappa = k12 * float(np.linalg.norm(A21, 2))
    Ar = A11 - A12 @ np.linalg.solve(A22, A21)
    nur, nuA = _lognorm(Ar), _lognorm(A)
    x0 = np.asarray(x0, float)
    x20n = float(np.linalg.norm(x0[k:]))
    B = float(np.linalg.norm(x0)) * max(1.0, math.exp(nuA * T))
    L = float(np.linalg.norm(A11, 2)) + kappa / mu
    G1 = math.expm1(nur * T) / nur if nur != 0 else T
    d = nur + mu
    G2 = (math.exp(nur * T) - math.exp(-mu * T)) / d if abs(d) > 1e-12 \
        else T * math.exp(nur * T)
    err = kappa * ((L * B + k12 * x20n) / mu**2 * G1 + B / mu * G2) \
        + k12 * x20n * G2
    xr = expm(Ar * T) @ x0[:k]
    return Certified(xr, float(err), Tier.RIGOROUS,
                     (f"mz-markov k={k} mu={mu:.3g} kappa={kappa:.3g} "
                      f"nur={nur:.3g} x20={x20n:.3g}",))


def conformal_closure(traj_full, traj_red, sampler, x_new,
                      n_cal: int = 99, rng=None) -> Certified:
    """Phase 4 rewrite, empirical tier: predict with the reduced model,
    certified by conformal calibration. Runs both models on n_cal i.i.d.
    initial conditions from sampler; err is the worst observed uniform
    (sup over time, 2-norm over state) deviation. For a fresh draw from
    the SAME distribution, P(deviation > err) <= 1/(n_cal+1) by
    exchangeability — a distribution-free guarantee, no model
    assumptions. Tier EMPIRICAL: valid for the sampled distribution,
    not for out-of-distribution initial conditions."""
    rng = np.random.default_rng(rng)

    def dev(x0):
        diff = np.asarray(traj_full(x0), float) - np.asarray(traj_red(x0), float)
        return float(np.max(np.linalg.norm(diff.reshape(len(diff), -1), axis=1)))

    worst = max(dev(sampler(rng)) for _ in range(n_cal))
    return Certified(traj_red(x_new), worst, Tier.EMPIRICAL,
                     (f"mz-conformal n_cal={n_cal} sup-t norm",),
                     fail_p=1.0 / (n_cal + 1))


def mz_search_slow(A: np.ndarray, x0: np.ndarray, T: float,
                   targets=(), tol: float = None):
    """Phase 4 rewrite: automatic slow-variable identification. Greedy
    search over which coordinates to resolve, scored by the certified
    closure error itself — "slow variables" are the split the machine
    certifies tightest. Starts from targets (coordinates the caller must
    keep); with none, seeds from the best single-or-pair split (pure
    greedy is myopic: a slow coordinate left in the fast sector kills the
    gap, and the structure only shows at pair level — deeper hidden
    structure than pairs would need a better search). Each step then adds
    the coordinate that most improves the certificate. With tol: stops at
    the smallest resolved set meeting
    it (the cheapest certified reduced model), raising if none does.
    Without: returns the best split found along the greedy path.
    Returns (Certified, slow_indices); value is x_slow(T) ordered as
    slow_indices. Splits without a fast-sector gap are skipped."""
    A = np.asarray(A, float)
    x0 = np.asarray(x0, float)
    n = len(x0)

    def evaluate(idx):
        perm = idx + [i for i in range(n) if i not in idx]
        try:
            return mz_closure_linear(A[np.ix_(perm, perm)], len(idx),
                                     x0[perm], T)
        except ValueError:
            return None

    slow = list(targets)
    best = (math.inf, None, None)
    if slow:
        c = evaluate(slow)
        if c is not None:
            if tol is not None and c.err <= tol:
                return c, slow
            best = (c.err, c, list(slow))
    else:
        # greedy from a single seed is myopic: one slow coordinate left in
        # the fast sector kills the gap, so the structure only shows at
        # pair level. Seed with the best single-or-pair split.
        for i in range(n):
            for j in range(-1, i):
                idx = [i] if j < 0 else [j, i]
                c = evaluate(idx)
                if c is not None and c.err < best[0]:
                    best = (c.err, c, idx)
        if best[1] is None:
            raise ValueError("no single or pair split admits a certified "
                             "closure (no spectral gap for any candidate)")
        slow = list(best[2])
        if tol is not None and best[0] <= tol:
            return best[1], slow
    while len(slow) < n - 1:
        step = None
        for i in range(n):
            if i in slow:
                continue
            c = evaluate(slow + [i])
            if c is not None and (step is None or c.err < step[0]):
                step = (c.err, c, slow + [i])
        if step is None:
            break                        # no candidate split has a gap
        slow = step[2]
        if step[0] < best[0]:
            best = step
        if tol is not None and step[0] <= tol:
            return step[1], step[2]
    if tol is not None:
        raise ValueError(f"no split certifies tol={tol:.3g}; best "
                         f"{best[0]:.3g} at coordinates {best[2]}")
    if best[1] is None:
        raise ValueError("no split admits a certified closure "
                         "(no spectral gap for any candidate)")
    return best[1], best[2]


# --------------------------------------------------------- CEM beachhead:
# certified 2D Helmholtz scattering (Lippmann-Schwinger, penetrable
# scatterer). The certified region is weak scattering: ||K|| < 1, with
# ||K|| bounded deterministically by Schur's ||K||_2 <= sqrt(||K||_1
# ||K||_inf) (the HMT probe tracks ||K||_F and is ~20x looser on this
# flat-spectrum oscillatory operator) — the bound's physical provenance
# is the scattering strength k^2 * contrast * area. The Neumann-series depth
# comes from the FAR-FIELD tolerance (query-first); the far-field
# functional propagates by Cauchy-Schwarz. Certificates are for the
# STATED discrete system (midpoint Nystrom, equal-area-disk self term);
# continuum discretization error is the named gap (asymptotic tier
# territory). Strong scattering needs resolvent-based certification —
# future work, refused today.


def _helmholtz_K(k, contrast, n, L):
    """Discrete Lippmann-Schwinger operator on an n x n midpoint grid over
    [-L/2, L/2]^2: K[i,j] = k^2 G(x_i, y_j) m(y_j) h^2, diagonal via the
    exact integral of G over the equal-area disk (a = h/sqrt(pi)):
    int_{|y|<a} (i/4) H0(k|y|) dy = (i pi / (2 k^2)) (ka H1(ka) + 2i/pi)."""
    from scipy.special import hankel1
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    pts = (X + 1j * Y).ravel()
    m = np.asarray(contrast(pts.real, pts.imag), complex)
    r = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(r, 1.0)
    G = 0.25j * hankel1(0, k * r)
    a = h / math.sqrt(math.pi)
    ka = k * a
    self_int = (1j * math.pi / (2 * k * k)) \
        * (ka * hankel1(1, ka) + 2j / math.pi)
    K = k * k * G * m[None, :] * h * h
    np.fill_diagonal(K, k * k * m * self_int)
    return pts, m, K


def helmholtz_scatter_farfield(k: float, contrast, n: int, L: float,
                               angles: np.ndarray, tol: float):
    """CEM rewrite: far-field pattern of a penetrable 2D scatterer under
    a unit plane wave along +x, each angle certified to |error| <= tol.
    Refuses outside the certified weak-scattering region (probe bound on
    ||K|| >= 1). Returns (Certified, stats); certificates are for the
    stated discrete system, exact-arithmetic."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    pts, m, K = _helmholtz_K(k, contrast, n, L)
    beta = math.sqrt(float(np.max(np.sum(np.abs(K), axis=0))
                           * np.max(np.sum(np.abs(K), axis=1))))
    if beta >= 1.0:
        raise ValueError(f"||K|| probe bound {beta:.3g} >= 1: outside the "
                         "certified weak-scattering (Born/Neumann) region")
    ui = np.exp(1j * k * pts.real)
    gamma = np.exp(1j * math.pi / 4) / math.sqrt(8 * math.pi * k)
    h2 = (L / n) ** 2
    W = gamma * k * k * m[None, :] * h2 * np.exp(
        -1j * k * (np.cos(angles)[:, None] * pts.real[None, :]
                   + np.sin(angles)[:, None] * pts.imag[None, :]))
    wnorm = np.linalg.norm(W, axis=1)
    # Neumann depth from the far-field tolerance: query-first
    u_target = tol / float(np.max(wnorm))
    ui_norm = float(np.linalg.norm(ui))
    N = max(0, math.ceil(math.log(u_target * (1 - beta) / ui_norm)
                         / math.log(beta) - 1))
    if N > 10_000:
        raise ValueError(f"tol={tol:.3g} needs {N} Neumann terms at "
                         f"||K||<={beta:.3g}: too close to the boundary")
    u, term = ui.copy(), ui.copy()
    for _ in range(N):
        term = K @ term
        u += term
    err_u = beta ** (N + 1) / (1 - beta) * ui_norm
    ff = W @ u
    angle_bound = wnorm * err_u
    stats = {"beta": beta, "n_terms": N, "err_u": err_u,
             "angle_bound": angle_bound}
    cert = Certified(ff, float(np.linalg.norm(angle_bound)), Tier.RIGOROUS,
                     (f"helmholtz-born k={k:g} beta={beta:.3g} N={N} "
                      "schur discrete-system scope",))
    return cert, stats


# ------------------------------------------------- multi-level butterfly:
# the genuine high-frequency rewrite. Complementary low-rank: sub-blocks
# K(t, s) with r_t * r_s ~ const have rank ~ k r_t r_s / D = O(1)
# uniformly, so a ladder that REFINES the row tree while COARSENING the
# column tree keeps every factor small: stage-0 factors touch the matrix
# once per column leaf; each stage recompresses the restriction of the
# previous bases to row children and records a small transfer matrix.
# Construction is heuristic (relative-tol SVD truncations); the
# certificate is A POSTERIORI on the assembled operator — probes of
# K w - B w — so validity never depends on the construction.


def _bisect_positions(pts, idx):
    """Positions splitting a cluster in two along its wider axis."""
    xs = pts[idx]
    key = xs.real if np.ptp(xs.real) >= np.ptp(xs.imag) else xs.imag
    order = np.argsort(key, kind="stable")
    return order[:len(idx) // 2], order[len(idx) // 2:]


def _quad_positions(pts, idx):
    """Positions splitting a cluster in four (two bisections): in 2D the
    DIAMETER must halve per butterfly level, which one binary split does
    not do — with branching 2 the stage rank products only shrink like
    2^(L/2) and every stage stays expensive (measured before this fix)."""
    a, b = _bisect_positions(pts, idx)
    a1, a2 = _bisect_positions(pts, idx[a])
    b1, b2 = _bisect_positions(pts, idx[b])
    return [a[a1], a[a2], b[b1], b[b2]]


class ButterflyBlock:
    """Butterfly factorization of one oscillatory kernel block. Guarantee
    per apply(q): ||value - K q||_2 <= beta ||q||_2 with the stated
    fail_p, beta certified a posteriori by probes on the assembled
    factorization (setup touches the dense block, like the H-matrix:
    the value is amortized applies). Tier RIGOROUS (exact arithmetic)."""

    def __init__(self, K, tgt, src, levels, eps=1e-4, n_probes=10, rng=None):
        rng = np.random.default_rng(rng)
        L, (m, n) = levels, K.shape
        if 4 ** L > min(m, n) // 8:
            raise ValueError(f"levels={L} too deep for a {m}x{n} block")
        tolc = eps      # per-factor; the a posteriori certificate absorbs
                        # accumulation, so no per-level safety split needed

        rowlvl, rowpos = [[np.arange(m)]], {}
        collvl = [[np.arange(n)]]
        for lv in range(L):
            nxt = []
            for i, t in enumerate(rowlvl[-1]):
                for c, p in enumerate(_quad_positions(tgt, t)):
                    rowpos[(lv + 1, 4 * i + c)] = p
                    nxt.append(t[p])
            rowlvl.append(nxt)
            collvl.append([s[p] for s in collvl[-1]
                           for p in _quad_positions(src, s)])

        def trunc_svd(A):
            U, sv, Vt = np.linalg.svd(A, full_matrices=False)
            keep = max(1, int(np.sum(sv > tolc * sv[0])))
            return U[:, :keep], sv[:keep, None] * Vt[:keep]

        self.C0, Us = [], {}
        for j, s in enumerate(collvl[L]):
            U, C = trunc_svd(K[:, s])
            Us[(0, j)] = U
            self.C0.append(C)
        self.E = {}
        for lv in range(L):
            new = {}
            for i2 in range(4 ** (lv + 1)):
                pos = rowpos[(lv + 1, i2)]
                for j2 in range(4 ** (L - lv - 1)):
                    G = np.hstack([Us[(i2 // 4, 4 * j2 + c)][pos]
                                   for c in range(4)])
                    Un, E = trunc_svd(G)
                    new[(i2, j2)] = Un
                    self.E[(lv + 1, i2, j2)] = E
            Us = new
        self.Ufin = [Us[(i, 0)] for i in range(4 ** L)]
        self.rows, self.cols, self.L = rowlvl[L], collvl[L], L
        self.m, self.dtype = m, np.result_type(K)
        self.apply_flops = sum(C.size for C in self.C0) \
            + sum(E.size for E in self.E.values()) \
            + sum(U.size for U in self.Ufin)

        # a posteriori certificate on the assembled operator
        W = rng.standard_normal((n, n_probes))
        resid = K @ W - np.stack([self._raw(W[:, i])
                                  for i in range(n_probes)], axis=1)
        scale = 10.0 * math.sqrt(2.0 / math.pi) \
            * (math.sqrt(2.0) if np.issubdtype(self.dtype,
                                               np.complexfloating) else 1.0)
        self.beta = scale * float(np.max(np.linalg.norm(resid, axis=0)))
        self.fail_p = 10.0 ** (-n_probes)

    def _raw(self, q):
        vec = {(0, j): C @ q[s] for j, (C, s)
               in enumerate(zip(self.C0, self.cols))}
        for lv in range(self.L):
            vec = {(i2, j2): self.E[(lv + 1, i2, j2)]
                   @ np.concatenate([vec[(i2 // 4, 4 * j2 + c)]
                                     for c in range(4)])
                   for i2 in range(4 ** (lv + 1))
                   for j2 in range(4 ** (self.L - lv - 1))}
        out = np.zeros(self.m, self.dtype)
        for i, t in enumerate(self.rows):
            out[t] = self.Ufin[i] @ vec[(i, 0)]
        return out

    def apply(self, q):
        value = self._raw(np.asarray(q))
        stats = {"apply_flops": self.apply_flops,
                 "dense_flops": self.m * len(q), "beta": self.beta}
        return Certified(value, self.beta * float(np.linalg.norm(q)),
                         Tier.RIGOROUS,
                         (f"butterfly L={self.L} beta={self.beta:.3g}",),
                         self.fail_p), stats


# ----------------------------------------------- chemistry energy bracket
# (TARGETS.md), matrix tier: a certified two-sided interval on the
# ground-state energy of a formable Hermitian H. Upper bound: Rayleigh
# quotient of a heuristic Lanczos vector — the variational theorem
# certifies it no matter where the vector came from. Lower bound:
# Cholesky feasibility proofs of H - cI >= 0 (the poor man's SDP dual
# certificate), bisected from a rigorous Gershgorin seed; a spurious FP
# Cholesky failure only loosens the bracket, never invalidates it.
# Exact-arithmetic tier (Cholesky success is FP-trusted, declared).
# The 2-RDM SDP lower bound — the version that scales past formable
# Hamiltonians — is the named next rung; so is a molecular-integrals
# pipeline. This demonstrates the bracket structure itself.


def eigen_bracket(H: np.ndarray, tol: float = None) -> Certified:
    """Certified bracket on lambda_min(H), H Hermitian: value +- err
    contains the true ground energy. Raises if tol is given and the
    achieved width exceeds it."""
    H = np.asarray(H)
    n = len(H)
    from scipy.sparse.linalg import eigsh
    try:
        _, V = eigsh(H, k=1, which="SA")
        v = V[:, 0]
    except Exception:                       # heuristic source; bracket
        v = np.ones(n)                      # stays valid, just looser
    v = v / np.linalg.norm(v)
    up = float(np.real(v.conj() @ (H @ v)))     # variational theorem

    def psd(c):
        try:
            np.linalg.cholesky(H - c * np.eye(n))
            return True
        except np.linalg.LinAlgError:
            return False

    d = np.real(np.diag(H))
    gersh = float(np.min(d - (np.sum(np.abs(H), axis=1) - np.abs(d))))
    r = float(np.linalg.norm(H @ v - up * v))
    lo = up - 2 * r - 1e-12 * (1 + abs(up))     # near-optimal guess ...
    if not psd(lo):
        lo = gersh                              # ... else rigorous seed
        if not psd(lo):
            raise ValueError("Gershgorin seed not certifiable (FP)")
    hi = up
    for _ in range(60):
        if hi - lo <= 1e-13 * (1 + abs(up)):
            break
        c = 0.5 * (lo + hi)
        if psd(c):
            lo = c
        else:
            hi = c
    # carry the FP margins: the Rayleigh quotient's evaluation error and
    # Cholesky's backward-stability slack (success proves PSD of
    # H - cI + E, ||E|| <~ n eps ||H||, Higham) — conservative constant
    pad = 8 * (n + 2) * np.finfo(float).eps \
        * (float(np.linalg.norm(H, 1)) + abs(up))
    up, lo = up + pad, lo - pad
    value, err = 0.5 * (up + lo), 0.5 * (up - lo)
    if tol is not None and err > tol:
        raise ValueError(f"bracket width {err:.3g} exceeds tol={tol:.3g} "
                         "(floating-point floor)")
    return Certified(value, err, Tier.RIGOROUS,
                     (f"eigen-bracket variational-upper cholesky-lower "
                      f"width={2 * err:.3g} +fp",))


# --------------------------------------- 2-RDM / marginal SDP lower bound,
# base rung: the block-marginal (Anderson) bound. Every length-ell window
# of the chain is bounded below independently, with per-bond weights
# 1/m_b (m_b = number of windows containing bond b) so the weighted
# windows sum EXACTLY to H; then E0 >= sum_w lambda_min(window), each
# lambda_min certified from below by eigen_bracket's Cholesky half.
# Correction multipliers ascend the SDP dual over the telescoping family:
# a Hermitian C on each window overlap, +C as the left part of one window
# and -C as the right part of its neighbor, so the sum still telescopes
# EXACTLY to H and any C yields a valid bound. lambda_min is concave in
# C; supergradient ascent (gradient = left minus right reduced density of
# the window ground state) climbs it, and every window is re-certified by
# Cholesky afterwards — the optimizer is pure quality. Richer correction
# structures than single-overlap operators remain (the fully general
# dual); this family already closes much of the gap. The
# upper half: a product of per-block Lanczos states — block energies and
# cross-block bond energies are explicit Rayleigh quotients, so the
# variational bound is rigorous whatever the vectors are. Cost 2^ell per
# distinct window, INDEPENDENT of N: the bracket scales past formable
# Hamiltonians.

_SPIN = (np.array([[0, 1], [1, 0]]) / 2,
         np.array([[0, -1j], [1j, 0]]) / 2,
         np.array([[1, 0], [0, -1]]) / 2)


def _heis_window(weights):
    """sum_j weights[j] * S_j . S_{j+1} on len(weights)+1 spins (dense)."""
    nsite = len(weights) + 1
    H = np.zeros((2 ** nsite, 2 ** nsite), complex)
    for pos, wgt in enumerate(weights):
        for s in _SPIN:
            op = np.eye(1)
            for j in range(nsite):
                op = np.kron(op, s if j in (pos, pos + 1) else np.eye(2))
            H += wgt * op
    return H.real


def _site_spin_expect(v, nsite, j):
    """(<Sx>, <Sz>) at site j for a real state v (<Sy> is exactly 0)."""
    out = []
    for s in (_SPIN[0], _SPIN[2]):
        op = np.eye(1)
        for i in range(nsite):
            op = np.kron(op, s if i == j else np.eye(2))
        out.append(float(v @ (op.real @ v)))
    return out


@functools.lru_cache(maxsize=None)
def _chain_correction(ell, iters):
    """Proximal-bundle ascent of the marginal-SDP dual over single-overlap
    corrections C. This family EXHAUSTS the fully general dual at fixed
    ell: translation invariance is WLOG in the bulk (shift-covariant
    constraints, linear objective — symmetrize any feasible point);
    non-consecutive overlap variables are redundant (their consistency is
    implied by consecutive pairs); and decomposition-weight freedom is
    absorbed (verified numerically: joint (C, weights) ascent reaches the
    identical plateau to six digits). The residual gap at fixed ell is
    the relaxation level itself — the hierarchy knob is ell.
    Bundle: lambda_min(W(C)) = min_v of exact affine cuts
    <v|Hw|v> + <rhoL(v) - rhoR(v), C>; master dual is a simplex QP over
    cut weights, solved by exponentiated gradient. iters = oracle calls
    (eigendecompositions); certification happens downstream, so this is
    pure quality."""
    Hw = _heis_window((1.0 / (ell - 1),) * (ell - 1))
    d = 2 ** (ell - 1)
    I2 = np.eye(2)

    def oracle(C):
        lam, V = np.linalg.eigh(Hw + np.kron(C, I2) - np.kron(I2, C))
        cuts = []
        for i in range(min(4, len(lam))):
            if lam[i] > lam[0] + 1e-7:
                break
            v = V[:, i]
            Vl, Vr = v.reshape(d, 2), v.reshape(2, d)
            cuts.append((float(v @ (Hw @ v)), Vl @ Vl.T - Vr.T @ Vr))
        return float(lam[0]), cuts

    Cref, tau = np.zeros((d, d)), 1.0
    fref, cuts = oracle(Cref)
    A = [a for a, _ in cuts]
    G = [g for _, g in cuts]
    best = (fref, Cref.copy())
    for _ in range(iters - 1):
        m = len(A)
        gram = np.array([[np.sum(G[i] * G[j]) for j in range(m)]
                         for i in range(m)])
        b = np.array([A[i] + np.sum(G[i] * Cref) for i in range(m)])
        mu = np.full(m, 1.0 / m)                    # simplex QP via EG
        eta = 1.0 / (1.0 + tau * float(np.max(np.abs(gram))))
        for _ in range(250):
            grad = b + tau * (gram @ mu)
            mu = mu * np.exp(-eta * (grad - grad @ mu))
            mu /= mu.sum()
        g = sum(w * Gi for w, Gi in zip(mu, G))
        Cnew = Cref + tau * g
        fnew, cuts = oracle(Cnew)
        for a, gi in cuts:
            A.append(a)
            G.append(gi)
        if len(A) > 60:                              # cap bundle size
            A, G = A[-60:], G[-60:]
        if fnew > fref:                              # serious step
            Cref, fref, tau = Cnew, fnew, min(tau * 1.4, 50.0)
        else:                                        # null step
            tau = max(tau * 0.6, 1e-3)
        if fref > best[0]:
            best = (fref, Cref.copy())
    return best[1]


def heisenberg_chain_bracket(N: int, ell: int = 8,
                             correction_iters: int = 80) -> Certified:
    """Certified two-sided bracket on the ground energy of the spin-1/2
    Heisenberg open chain of N sites, at cost 2^ell independent of N.
    correction_iters=0 disables the SDP-dual multiplier ascent."""
    if N <= ell:
        c = eigen_bracket(_heis_window((1.0,) * (N - 1)))
        return replace(c, provenance=(f"chain-bracket exact N={N}",))
    C = _chain_correction(ell, correction_iters) if correction_iters else None
    # lower: weighted sliding windows summing exactly to H, plus exactly
    # telescoping corrections (+C on a window's left overlap, -C on its
    # neighbour's right overlap)
    m = [min(N - ell, i) - max(0, i - ell + 2) + 1 for i in range(N - 1)]
    cache = {}
    lower = 0.0
    for w in range(N - ell + 1):
        key = (tuple(1.0 / m[w + j] for j in range(ell - 1)),
               w >= 1, w <= N - ell - 1)
        if key not in cache:
            Hw = _heis_window(key[0])
            if C is not None and key[1]:
                Hw = Hw + np.kron(C, np.eye(2))
            if C is not None and key[2]:
                Hw = Hw - np.kron(np.eye(2), C)
            c = eigen_bracket(Hw)
            cache[key] = c.value - c.err
        lower += cache[key]
    # upper: product of per-block Lanczos states; all terms are explicit
    # Rayleigh quotients of the product state
    from scipy.sparse.linalg import eigsh
    sizes = [ell] * (N // ell)
    if N % ell == 1:
        sizes[-1] += 1
    elif N % ell:
        sizes.append(N % ell)
    upper, prev_edge, block = 0.0, None, {}
    for size in sizes:
        if size not in block:
            Hb = _heis_window((1.0,) * (size - 1))
            _, V = eigsh(Hb, k=1, which="SA")
            v = V[:, 0] / np.linalg.norm(V[:, 0])
            e = float(v @ (Hb @ v))
            pad = 8 * (2 ** size + 2) * np.finfo(float).eps \
                * (float(np.linalg.norm(Hb, 1)) + abs(e))
            block[size] = (e + pad, _site_spin_expect(v, size, 0),
                           _site_spin_expect(v, size, size - 1))
        e, left, right = block[size]
        upper += e
        if prev_edge is not None:     # cross-block bond <S>.<S>
            upper += prev_edge[0] * left[0] + prev_edge[1] * left[1]
        prev_edge = right
    return Certified(0.5 * (upper + lower), 0.5 * (upper - lower),
                     Tier.RIGOROUS,
                     (f"chain-bracket marginal-lower ell={ell} N={N} "
                      f"iters={correction_iters} product-upper",))


# ------------------------------------------- molecular-integrals pipeline:
# the energy bracket on an actual molecule. H2 in STO-3G needs only
# s-type Gaussian integrals, which have complete closed forms via the
# Boys function — fully self-contained (no quantum-chemistry package;
# the only external data is the published STO-3G hydrogen basis,
# Hehre-Stewart-Pople). Second quantization via Jordan-Wigner on 4 spin
# orbitals gives a 16x16 Fock-space Hamiltonian — bracketing it certifies
# the ground energy over ALL particle-number sectors at once, and for H2
# near equilibrium that sector is the neutral molecule. Verified three
# ways in the tests: integrals vs 3D grid quadrature, dissociation vs
# two isolated atoms, and an independent 2x2 MO-basis CI assembly
# (exact FCI for this system by parity).


_STO3G_H_RAW = ((3.42525091, 0.15432897), (0.62391373, 0.53532814),
                (0.16885540, 0.44463454))     # Hehre-Stewart-Pople


def _sto3g_h():
    """(exponent, contraction * primitive norm) for STO-3G hydrogen."""
    return tuple((a, c * (2 * a / math.pi) ** 0.75) for a, c in _STO3G_H_RAW)


def _boys0(t):
    return 1.0 if t < 1e-12 else 0.5 * math.sqrt(math.pi / t) \
        * math.erf(math.sqrt(t))


def _h2_integrals(R):
    """(S, h, eri, enuc) for H2 at bond length R bohr in STO-3G:
    2x2 overlap and core Hamiltonian, chemists'-notation (pq|rs) ERI."""
    centers = [np.zeros(3), np.array([0.0, 0.0, R])]
    prims = _sto3g_h()
    S = np.zeros((2, 2))
    h = np.zeros((2, 2))
    for i, A in enumerate(centers):
        for j, B in enumerate(centers):
            ab2 = float(np.dot(A - B, A - B))
            for a, ca in prims:
                for b, cb in prims:
                    p = a + b
                    mu = a * b / p
                    K = math.exp(-mu * ab2)
                    P = (a * A + b * B) / p
                    S[i, j] += ca * cb * (math.pi / p) ** 1.5 * K
                    h[i, j] += ca * cb * mu * (3 - 2 * mu * ab2) \
                        * (math.pi / p) ** 1.5 * K
                    for C in centers:            # nuclear attraction, Z=1
                        pc2 = float(np.dot(P - C, P - C))
                        h[i, j] -= ca * cb * (2 * math.pi / p) * K \
                            * _boys0(p * pc2)
    eri = np.zeros((2, 2, 2, 2))
    for i, A in enumerate(centers):
        for j, B in enumerate(centers):
            ab2 = float(np.dot(A - B, A - B))
            for k, C in enumerate(centers):
                for l, D in enumerate(centers):
                    cd2 = float(np.dot(C - D, C - D))
                    val = 0.0
                    for a, ca in prims:
                        for b, cb in prims:
                            p = a + b
                            Kab = math.exp(-a * b / p * ab2)
                            P = (a * A + b * B) / p
                            for c, cc in prims:
                                for d, cd in prims:
                                    q = c + d
                                    Kcd = math.exp(-c * d / q * cd2)
                                    Q = (c * C + d * D) / q
                                    pq2 = float(np.dot(P - Q, P - Q))
                                    val += ca * cb * cc * cd \
                                        * 2 * math.pi ** 2.5 \
                                        / (p * q * math.sqrt(p + q)) \
                                        * Kab * Kcd \
                                        * _boys0(p * q / (p + q) * pq2)
                    eri[i, j, k, l] = val
    return S, h, eri, 1.0 / R


def hydrogen_atom_energy():
    """<phi|T + V|phi> for a single STO-3G hydrogen atom (the
    dissociation reference), from the same closed forms."""
    prims = _sto3g_h()
    e = s = 0.0
    for a, ca in prims:
        for b, cb in prims:
            p = a + b
            s += ca * cb * (math.pi / p) ** 1.5
            e += ca * cb * (3 * a * b / p * (math.pi / p) ** 1.5
                            - 2 * math.pi / p)
    return e / s


def sto3g_h2_hamiltonian(R):
    """16x16 Fock-space Hamiltonian for H2/STO-3G at bond length R bohr
    (Jordan-Wigner, nuclear repulsion included). The AO basis is first
    Lowdin-orthogonalized so creation operators are fermionic."""
    S, hmat, eri, enuc = _h2_integrals(R)
    w, U = np.linalg.eigh(S)
    Xo = U @ np.diag(w ** -0.5) @ U.T           # Lowdin S^(-1/2)
    hmat = Xo @ hmat @ Xo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, Xo, Xo, Xo, Xo)
    # Jordan-Wigner ladder operators on 4 spin orbitals (2*spatial+spin)
    I2, Zm = np.eye(2), np.diag([1.0, -1.0])
    low = np.array([[0.0, 1.0], [0.0, 0.0]])
    ann = []
    for pp in range(4):
        mats = [Zm] * pp + [low] + [I2] * (3 - pp)
        op = mats[0]
        for mm in mats[1:]:
            op = np.kron(op, mm)
        ann.append(op)
    H = enuc * np.eye(16)
    for p in range(2):
        for q in range(2):
            for sp in range(2):
                H += hmat[p, q] * ann[2 * p + sp].T @ ann[2 * q + sp]
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s2 in range(2):
                    for sa in range(2):
                        for sb in range(2):
                            P, Q = 2 * p + sa, 2 * q + sb
                            Rr, Ss = 2 * r + sa, 2 * s2 + sb
                            H += 0.5 * eri[p, r, q, s2] \
                                * ann[P].T @ ann[Q].T @ ann[Ss] @ ann[Rr]
    return H


def h2_energy_bracket(R: float) -> Certified:
    """Certified two-sided bracket on the H2/STO-3G total ground energy
    (hartree) at bond length R bohr — over all particle-number sectors."""
    c = eigen_bracket(sto3g_h2_hamiltonian(R))
    return replace(c, provenance=(f"h2-sto3g-bracket R={R:g} "
                                  + c.provenance[0],))


# --------------------------------------- McMurchie-Davidson recursions:
# integrals for arbitrary Cartesian angular momentum, enabling p (and
# higher) orbitals. Hermite expansion coefficients E by recursion,
# Hermite Coulomb integrals R on a general Boys function (series +
# stable downward recursion; asymptotic branch for large t). Verified
# in tests three ways: l=0 reproduces the closed-form s-integrals to
# machine precision (the previous pipeline is the oracle), genuine
# p-integrals match 3D grid quadrature, and the p-polarized H2 bracket
# lies strictly below the s-only bracket — a theorem about two
# certified intervals.


def _boys(nmax, t):
    """F_0..F_nmax(t): series at nmax + stable downward recursion;
    asymptotic upward branch for t > 35 (e^-t below double precision)."""
    if t > 35.0:
        F = [0.5 * math.sqrt(math.pi / t)]
        for n in range(1, nmax + 1):
            F.append(F[-1] * (2 * n - 1) / (2 * t))
        return F
    s, term, k = 0.0, 1.0 / (2 * nmax + 1), 0
    while k < 250:
        s += term
        k += 1
        term *= 2 * t / (2 * nmax + 2 * k + 1)
        if term < 1e-17 * s:
            break
    F = [0.0] * (nmax + 1)
    F[nmax] = s * math.exp(-t)
    for n in range(nmax, 0, -1):
        F[n - 1] = (2 * t * F[n] + math.exp(-t)) / (2 * n - 1)
    return F


def _md_E(i, j, t, Q, a, b, memo):
    """Hermite expansion coefficient E_t^{ij} (1D), Helgaker recursion."""
    if t < 0 or t > i + j:
        return 0.0
    key = (i, j, t)
    if key in memo:
        return memo[key]
    p = a + b
    if i == j == t == 0:
        val = math.exp(-a * b / p * Q * Q)
    elif i > 0:
        val = _md_E(i - 1, j, t - 1, Q, a, b, memo) / (2 * p) \
            + (-b * Q / p) * _md_E(i - 1, j, t, Q, a, b, memo) \
            + (t + 1) * _md_E(i - 1, j, t + 1, Q, a, b, memo)
    else:
        val = _md_E(i, j - 1, t - 1, Q, a, b, memo) / (2 * p) \
            + (a * Q / p) * _md_E(i, j - 1, t, Q, a, b, memo) \
            + (t + 1) * _md_E(i, j - 1, t + 1, Q, a, b, memo)
    memo[key] = val
    return val


def _md_R(t, u, v, n, p, PC, F, memo):
    """Hermite Coulomb integral R^n_{tuv}, Helgaker recursion."""
    if t < 0 or u < 0 or v < 0:
        return 0.0
    key = (t, u, v, n)
    if key in memo:
        return memo[key]
    if t == u == v == 0:
        val = (-2.0 * p) ** n * F[n]
    elif t > 0:
        val = (t - 1) * _md_R(t - 2, u, v, n + 1, p, PC, F, memo) \
            + PC[0] * _md_R(t - 1, u, v, n + 1, p, PC, F, memo)
    elif u > 0:
        val = (u - 1) * _md_R(t, u - 2, v, n + 1, p, PC, F, memo) \
            + PC[1] * _md_R(t, u - 1, v, n + 1, p, PC, F, memo)
    else:
        val = (v - 1) * _md_R(t, u, v - 2, n + 1, p, PC, F, memo) \
            + PC[2] * _md_R(t, u, v - 1, n + 1, p, PC, F, memo)
    memo[key] = val
    return val


def _cart_norm(a, l):
    """Norm of the primitive x^i y^j z^k exp(-a r^2)."""
    df = lambda n: math.prod(range(2 * n - 1, 0, -2)) if n else 1
    return (2 * a / math.pi) ** 0.75 * (4 * a) ** (sum(l) / 2) \
        / math.sqrt(df(l[0]) * df(l[1]) * df(l[2]))


def _md_integrals(atoms, shells):
    """(S, h, eri, enuc) for Cartesian-Gaussian AOs of arbitrary angular
    momentum. atoms: [(Z, xyz)]; shells: one AO each, (xyz, (i,j,k),
    [(exponent, contraction), ...]). Contractions are multiplied by
    primitive norms here; overall AO normalization is NOT assumed
    (Lowdin handles any overlap — the S11 != 1 lesson)."""
    nao = len(shells)
    aos = []
    for center, l, prims in shells:
        aos.append((np.asarray(center, float), l,
                    [(a, c * _cart_norm(a, l)) for a, c in prims]))

    def pair_terms(ao1, ao2):
        A, la, pa = ao1
        B, lb, pb = ao2
        for a, ca in pa:
            for b, cb in pb:
                yield a, b, ca * cb, A, la, B, lb

    def s1d(i, j, Q, a, b):
        return _md_E(i, j, 0, Q, a, b, {}) * math.sqrt(math.pi / (a + b))

    S = np.zeros((nao, nao))
    h = np.zeros((nao, nao))
    for m in range(nao):
        for n_ in range(nao):
            for a, b, cc, A, la, B, lb in pair_terms(aos[m], aos[n_]):
                sx = [s1d(la[d], lb[d], A[d] - B[d], a, b) for d in range(3)]
                S[m, n_] += cc * sx[0] * sx[1] * sx[2]
                tx = []
                for d in range(3):
                    j = lb[d]
                    t = -2 * b * b * s1d(la[d], j + 2, A[d] - B[d], a, b) \
                        + b * (2 * j + 1) * sx[d]
                    if j >= 2:
                        t -= 0.5 * j * (j - 1) \
                            * s1d(la[d], j - 2, A[d] - B[d], a, b)
                    tx.append(t)
                h[m, n_] += cc * (tx[0] * sx[1] * sx[2]
                                  + sx[0] * tx[1] * sx[2]
                                  + sx[0] * sx[1] * tx[2])
                p = a + b
                P = (a * A + b * B) / p
                Ex = [[_md_E(la[d], lb[d], t, A[d] - B[d], a, b, {})
                       for t in range(la[d] + lb[d] + 1)] for d in range(3)]
                ntot = sum(la) + sum(lb)
                for Z, C in atoms:
                    PC = P - np.asarray(C, float)
                    F = _boys(ntot, p * float(PC @ PC))
                    memo = {}
                    val = 0.0
                    for t in range(la[0] + lb[0] + 1):
                        for u in range(la[1] + lb[1] + 1):
                            for v in range(la[2] + lb[2] + 1):
                                val += Ex[0][t] * Ex[1][u] * Ex[2][v] \
                                    * _md_R(t, u, v, 0, p, PC, F, memo)
                    h[m, n_] -= cc * Z * 2 * math.pi / p * val

    eri = np.zeros((nao, nao, nao, nao))
    for m in range(nao):
        for n_ in range(nao):
            for k in range(nao):
                for l_ in range(nao):
                    val = 0.0
                    for a, b, cab, A, la, B, lb in pair_terms(aos[m], aos[n_]):
                        p = a + b
                        P = (a * A + b * B) / p
                        Eab = [[_md_E(la[d], lb[d], t, A[d] - B[d], a, b, {})
                                for t in range(la[d] + lb[d] + 1)]
                               for d in range(3)]
                        for c, d_, ccd, Cc, lc, D, ld in pair_terms(
                                aos[k], aos[l_]):
                            q = c + d_
                            Q = (c * Cc + d_ * D) / q
                            alpha = p * q / (p + q)
                            PQ = P - Q
                            Ecd = [[_md_E(lc[d], ld[d], t, Cc[d] - D[d],
                                          c, d_, {})
                                    for t in range(lc[d] + ld[d] + 1)]
                                   for d in range(3)]
                            ntot = sum(la) + sum(lb) + sum(lc) + sum(ld)
                            F = _boys(ntot, alpha * float(PQ @ PQ))
                            memo = {}
                            acc = 0.0
                            for t in range(la[0] + lb[0] + 1):
                                for u in range(la[1] + lb[1] + 1):
                                    for v in range(la[2] + lb[2] + 1):
                                        e1 = Eab[0][t] * Eab[1][u] * Eab[2][v]
                                        if e1 == 0.0:
                                            continue
                                        for t2 in range(lc[0] + ld[0] + 1):
                                            for u2 in range(lc[1] + ld[1] + 1):
                                                for v2 in range(
                                                        lc[2] + ld[2] + 1):
                                                    e2 = Ecd[0][t2] \
                                                        * Ecd[1][u2] \
                                                        * Ecd[2][v2]
                                                    if e2 == 0.0:
                                                        continue
                                                    acc += e1 * e2 * (-1) ** (
                                                        t2 + u2 + v2) \
                                                        * _md_R(t + t2,
                                                                u + u2,
                                                                v + v2, 0,
                                                                alpha, PQ,
                                                                F, memo)
                            val += cab * ccd * 2 * math.pi ** 2.5 \
                                / (p * q * math.sqrt(p + q)) * acc
                    eri[m, n_, k, l_] = val
    enuc = sum(atoms[i][0] * atoms[j][0]
               / float(np.linalg.norm(np.asarray(atoms[i][1])
                                      - np.asarray(atoms[j][1])))
               for i in range(len(atoms)) for j in range(i + 1, len(atoms)))
    return S, h, eri, enuc


def _jw_ann(nso):
    """Sparse Jordan-Wigner annihilation operators on nso spin orbitals."""
    from scipy import sparse
    I2 = sparse.identity(2, format="csr")
    Zm = sparse.diags([1.0, -1.0]).tocsr()
    low = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    ann = []
    for pp in range(nso):
        op = sparse.identity(1, format="csr")
        for mm in [Zm] * pp + [low] + [I2] * (nso - pp - 1):
            op = sparse.kron(op, mm, format="csr")
        ann.append(op)
    return ann


def _fock_hamiltonian(S, h, eri, enuc):
    """Lowdin-orthogonalize, Jordan-Wigner, assemble the 4^nao Fock-space
    Hamiltonian (sparse internally, dense out)."""
    from scipy import sparse
    w, U = np.linalg.eigh(S)
    Xo = U @ np.diag(w ** -0.5) @ U.T
    h = Xo @ h @ Xo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, Xo, Xo, Xo, Xo)
    nso = 2 * len(h)
    ann = _jw_ann(nso)
    dim = 2 ** nso
    H = sparse.identity(dim, format="csr") * enuc
    n_sp = len(h)
    for p in range(n_sp):
        for q in range(n_sp):
            if h[p, q] == 0.0:
                continue
            for sp in range(2):
                H = H + h[p, q] * (ann[2 * p + sp].T @ ann[2 * q + sp])
    for p in range(n_sp):
        for q in range(n_sp):
            for r in range(n_sp):
                for s2 in range(n_sp):
                    g = eri[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    for sa in range(2):
                        for sb in range(2):
                            H = H + 0.5 * g * (
                                ann[2 * p + sa].T @ ann[2 * q + sb].T
                                @ ann[2 * s2 + sb] @ ann[2 * r + sa])
    return np.asarray(H.todense())


def h2_polarized_bracket(R: float) -> Certified:
    """Certified bracket for H2 with an s + pz(alpha=1.1) basis on each
    atom (sigma polarization; the MD engine handles all components,
    tested by quadrature). 256-dim Fock space."""
    atoms = [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, R))]
    pz = ((1.1, 1.0),)
    shells = [((0.0, 0.0, 0.0), (0, 0, 0), _STO3G_H_RAW),
              ((0.0, 0.0, R), (0, 0, 0), _STO3G_H_RAW),
              ((0.0, 0.0, 0.0), (0, 0, 1), pz),
              ((0.0, 0.0, R), (0, 0, 1), pz)]
    c = eigen_bracket(_fock_hamiltonian(*_md_integrals(atoms, shells)))
    return replace(c, provenance=(f"h2-s+pz-bracket R={R:g} "
                                  + c.provenance[0],))


# ---------------------------------------------------------- the marriage:
# molecular integrals x marginal-SDP bracket, on hydrogen chains — a
# certified two-sided bracket for molecular Fock spaces too big to form.
# Terms are classified by orbital spread (JW strings stay inside
# contiguous windows, so window lambda_min bounds are valid for
# fermions). The genuinely molecular difficulty — long-range Coulomb —
# is handled exactly per far atom pair: F_ij = g(n_i-1)(n_j-1)
# + (v+g)n_i + (v+g)n_j + (1/R-g), with the quadratic part bounded by
# operator AM-GM, g(n_i-1)(n_j-1) >= -(g/2)[(n_i-1)^2 + (n_j-1)^2] —
# local charge-fluctuation penalties absorbed into windows; linear parts
# and constants exact. Remaining far terms use CAUCHY-SCHWARZ ABSORPTION
# (g XY + h.c. >= -|g|(XX' + Y'Y)): the JW Z-strings are unitary and
# vanish inside XX', which for distinct modes is a pure occupation
# product (creation -> n, annihilation -> 1-n; reversed for Y'Y) — so
# every far term absorbs into diagonal occupation products on its two
# compact sides, local to windows, with small ground-state expectations
# replacing the flat norms that used to dominate the gap (1.04 of
# 1.79 Ha at H6/ell=3; the absorption recovered ~0.52). Repeated-mode
# or window-overflowing sides fall back to flat bounds (rare, tiny).
# Shared-C window multipliers below add a further modest tightening.
# The balanced-eps outer loop (cs_rounds) is monotone-safe by
# best-tracking; measured on H-chains its optimum is eps=1 — the
# balancing family is exhausted at the start here, and the naive
# greedy update would LOSE 5 mHa/atom (not an ascent step). Upper: product of exactly solved atom blocks, cross energies
# by exact factorization of block-diagonal 1-RDMs (fermionic signs are
# benign — cross operators move in even pairs). No correction
# multipliers yet (the Heisenberg bundle machinery is the named
# tightening path).


@functools.lru_cache(maxsize=None)
def _h_chain_basis(n, d):
    """Lowdin-orthogonalized (T, V[c], eri, enuc) for the n-atom hydrogen
    chain at spacing d bohr, STO-3G. V[c] is the attraction to nucleus c
    separately — the far-pair decomposition needs it."""
    centers = [np.array([0.0, 0.0, i * d]) for i in range(n)]
    prims = _sto3g_h()
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i, n):
            A, B = centers[i], centers[j]
            ab2 = float(np.dot(A - B, A - B))
            for a, ca in prims:
                for b, cb in prims:
                    p = a + b
                    mu = a * b / p
                    K = math.exp(-mu * ab2)
                    P = (a * A + b * B) / p
                    S[i, j] += ca * cb * (math.pi / p) ** 1.5 * K
                    T[i, j] += ca * cb * mu * (3 - 2 * mu * ab2) \
                        * (math.pi / p) ** 1.5 * K
                    for c in range(n):
                        pc2 = float(np.dot(P - centers[c], P - centers[c]))
                        V[c, i, j] -= ca * cb * (2 * math.pi / p) * K \
                            * _boys0(p * pc2)
            S[j, i] = S[i, j]
            T[j, i] = T[i, j]
            V[:, j, i] = V[:, i, j]
    eri = np.zeros((n, n, n, n))
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    for pi, (i, j) in enumerate(pairs):
        for k, l in pairs[pi:]:
            A, B, C, D = centers[i], centers[j], centers[k], centers[l]
            ab2 = float(np.dot(A - B, A - B))
            cd2 = float(np.dot(C - D, C - D))
            val = 0.0
            for a, ca in prims:
                for b, cb in prims:
                    p = a + b
                    Kab = math.exp(-a * b / p * ab2)
                    P = (a * A + b * B) / p
                    for c, cc in prims:
                        for dd, cd_ in prims:
                            q = c + dd
                            Kcd = math.exp(-c * dd / q * cd2)
                            Q = (c * C + dd * D) / q
                            pq2 = float(np.dot(P - Q, P - Q))
                            val += ca * cb * cc * cd_ \
                                * 2 * math.pi ** 2.5 \
                                / (p * q * math.sqrt(p + q)) * Kab * Kcd \
                                * _boys0(p * q / (p + q) * pq2)
            for a_, b_ in ((i, j), (j, i)):
                for c_, d_ in ((k, l), (l, k)):
                    eri[a_, b_, c_, d_] = eri[c_, d_, a_, b_] = val
    w, U = np.linalg.eigh(S)
    Xo = U @ np.diag(w ** -0.5) @ U.T
    T = Xo @ T @ Xo
    V = np.einsum("cpq,pi,qj->cij", V, Xo, Xo)
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, Xo, Xo, Xo, Xo)
    enuc = sum(1.0 / (d * (j - i)) for i in range(n) for j in range(i + 1, n))
    return T, V, eri, enuc


def h_chain_fock_hamiltonian(n, d):
    """Dense 4^n Fock Hamiltonian for the hydrogen chain (formable n)."""
    T, V, eri, enuc = _h_chain_basis(n, d)
    return _fock_hamiltonian(np.eye(n), T + V.sum(0), eri, enuc)


def _window_operator(hw, eriw, lin, quad, const, extras=()):
    """Sparse-assembled window operator on 2*len(hw) spin orbitals:
    one- and two-electron parts, per-site linear n and -(quad)(n-1)^2
    charge penalties, a constant, and diagonal occupation-product extras
    [(coef, ((local_spin_orbital, occupied_bool), ...)), ...] from the
    Cauchy-Schwarz far-term absorption."""
    from scipy import sparse
    nsp = len(hw)
    ann = _jw_ann(2 * nsp)
    dim = 4 ** nsp
    H = sparse.identity(dim, format="csr") * const
    for p in range(nsp):
        for q in range(nsp):
            if hw[p, q] != 0.0:
                for sp in range(2):
                    H = H + hw[p, q] * (ann[2 * p + sp].T @ ann[2 * q + sp])
    for p in range(nsp):
        for q in range(nsp):
            for r in range(nsp):
                for s2 in range(nsp):
                    g = eriw[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    for sa in range(2):
                        for sb in range(2):
                            H = H + 0.5 * g * (
                                ann[2 * p + sa].T @ ann[2 * q + sb].T
                                @ ann[2 * s2 + sb] @ ann[2 * r + sa])
    eye = sparse.identity(dim, format="csr")
    for i in range(nsp):
        num = ann[2 * i].T @ ann[2 * i] + ann[2 * i + 1].T @ ann[2 * i + 1]
        if lin[i] != 0.0:
            H = H + lin[i] * num
        if quad[i] != 0.0:
            dev = num - eye
            H = H - quad[i] * (dev @ dev)
    if extras:
        ndiag = [np.asarray((ann[q].T @ ann[q]).todense()).diagonal()
                 for q in range(2 * nsp)]
        acc = np.zeros(dim)
        for coef, pattern in extras:
            dv = np.full(dim, coef)
            for so, occ in pattern:
                dv = dv * (ndiag[so] if occ else 1.0 - ndiag[so])
            acc += dv
        H = H + sparse.diags(acc)
    return np.asarray(H.todense())


def _ground_vec(M):
    """Ground eigenpair; Lanczos above small dims (only the vector is
    needed by the multiplier oracle and the eps loop)."""
    if len(M) < 512:
        lam, Vv = np.linalg.eigh(M)
        return lam[0], Vv[:, 0]
    from scipy.sparse.linalg import eigsh
    lam, Vv = eigsh(M, k=1, which="SA")
    return float(lam[0]), Vv[:, 0]


def _window_multipliers(mats, D, iters):
    """Proximal-bundle ascent of sum_w lambda_min over PER-OVERLAP
    Hermitian corrections C_1..C_{nw-1}: window w gains +C_w x I on its
    left overlap (w >= 1) and -I x C_{w+1} on its right (w <= nw-2), so
    the sum telescopes exactly at the qubit level for any C's — validity
    never constrains the optimizer. Jointly concave; exact affine cuts
    from window ground vectors (each cuts into the two C's touching it).
    Shared-C is the diagonal of this family; per-overlap freedom is
    where the finite chain's edge effects live. Returns [C_w]."""
    dim, nw = len(mats[0]), len(mats)
    E = dim // D
    IE = np.eye(E)
    nov = nw - 1

    def unflat(x):
        return [x[k * D * D:(k + 1) * D * D].reshape(D, D)
                for k in range(nov)]

    def build(w, Cs):
        M = mats[w]
        if w >= 1:
            M = M + np.kron(Cs[w - 1], IE)
        if w <= nw - 2:
            M = M - np.kron(IE, Cs[w])
        return M

    def oracle(x):
        Cs = unflat(x)
        tot, const = 0.0, 0.0
        G = [np.zeros((D, D)) for _ in range(nov)]
        for w in range(nw):
            lam0, v = _ground_vec(build(w, Cs))
            tot += lam0
            const += float(v @ (mats[w] @ v))
            Vl, Vr = v.reshape(D, E), v.reshape(E, D)
            if w >= 1:
                G[w - 1] += Vl @ Vl.T
            if w <= nw - 2:
                G[w] -= Vr.T @ Vr
        return tot, const, np.concatenate([g.ravel() for g in G])

    xref, tau = np.zeros(nov * D * D), 1.0
    fref, a0, g0 = oracle(xref)
    A, G = [a0], [g0]
    gram = np.array([[float(g0 @ g0)]])   # maintained incrementally
    best = (fref, xref.copy())
    for _ in range(iters - 1):
        m = len(A)
        b = np.array([A[i] + float(G[i] @ xref) for i in range(m)])
        mu = np.full(m, 1.0 / m)
        eta = 1.0 / (1.0 + tau * float(np.max(np.abs(gram))))
        for _ in range(250):
            grad = b + tau * (gram @ mu)
            mu = mu * np.exp(-eta * (grad - grad @ mu))
            mu /= mu.sum()
        g = sum(w_ * Gi for w_, Gi in zip(mu, G))
        xnew = xref + tau * g
        fnew, an, gn = oracle(xnew)
        A.append(an)
        G.append(gn)
        row = np.array([float(gn @ Gi) for Gi in G])
        gram = np.block([[gram, row[:-1, None]], [row[None, :-1],
                                                  np.array([[row[-1]]])]])
        if len(A) > 40:                    # cut vectors are large at
            A, G = A[-40:], G[-40:]        # big D; cap the bundle
            gram = gram[-40:, -40:]
        if fnew > fref:
            xref, fref, tau = xnew, fnew, min(tau * 1.4, 50.0)
        else:
            tau = max(tau * 0.6, 1e-3)
        if fref > best[0]:
            best = (fref, xref.copy())
    return unflat(best[1])


def h_chain_bracket(n: int, d: float = 1.8, ell: int = 3,
                    correction_iters: int = 60,
                    cs_rounds: int = 2) -> Certified:
    """Certified two-sided bracket on the ground energy of the n-atom
    hydrogen chain (STO-3G, spacing d bohr), at window cost 4^ell.
    correction_iters=0 disables the shared-multiplier bundle ascent."""
    T, V, eri, _ = _h_chain_basis(n, d)
    h_full = T + V.sum(0)

    def mcount(lo, hi):
        return min(lo, n - ell) - max(0, hi - ell + 1) + 1

    nw = n - ell + 1
    hW = [np.zeros((ell, ell)) for _ in range(nw)]
    eriW = [np.zeros((ell, ell, ell, ell)) for _ in range(nw)]
    linW = [np.zeros(ell) for _ in range(nw)]
    quadW = [np.zeros(ell) for _ in range(nw)]
    constW = [0.0] * nw
    lower_const, penalty = 0.0, 0.0

    cs_terms = []      # (coef, left pattern, right pattern): absorbed as
                       # -coef*(eps*L + R/eps), eps tuned by the outer loop

    def side_fits(modes):
        atoms = [m[0] // 2 for m in modes]
        return max(atoms) - min(atoms) < ell

    def windows_of(lo, hi):
        return range(max(0, hi - ell + 1), min(lo, n - ell) + 1)

    # one-electron terms
    for i in range(n):
        for j in range(n):
            lo, hi = min(i, j), max(i, j)
            if hi - lo < ell:
                hij = h_full[i, j]
                if i == j:      # far attractions leave for the F pairs
                    hij -= sum(V[c, i, i] for c in range(n)
                               if abs(c - i) >= ell)
                m = mcount(lo, hi)
                for w in windows_of(lo, hi):
                    hW[w][i - w, j - w] += hij / m
            elif i < j:
                # far hopping: h(a'a + h.c.) >= -|h|(eps n_i + n_j/eps)
                # per spin, eps tuned by the outer loop
                for sp in range(2):
                    cs_terms.append((abs(h_full[i, j]),
                                     ((2 * i + sp, True),),
                                     ((2 * j + sp, True),)))
    # two-electron terms, enumerated exactly as the assembly does
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s2 in range(n):
                    g = eri[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    lo, hi = min(p, q, r, s2), max(p, q, r, s2)
                    if hi - lo < ell:
                        m = mcount(lo, hi)
                        for w in windows_of(lo, hi):
                            eriW[w][p - w, r - w, q - w, s2 - w] += g / m
                    elif p == r and q == s2 and abs(p - q) >= ell:
                        pass    # the direct far pair, handled below
                    elif (r, s2, p, q) < (p, q, r, s2):
                        pass    # h.c. partner: processed once, below
                    else:
                        # Cauchy-Schwarz absorption: split the term's modes
                        # at its largest atom gap; Z-strings vanish in XX',
                        # which for distinct modes is a pure occupation
                        # product (creation -> n, annihilation -> 1-n; the
                        # reverse for Y'Y). |g| only — no sign bookkeeping.
                        atoms = sorted({p, q, r, s2})
                        gaps = [atoms[k + 1] - atoms[k]
                                for k in range(len(atoms) - 1)]
                        cut = atoms[int(np.argmax(gaps))]
                        ok = True
                        for sa in range(2):
                            for sb in range(2):
                                modes = [(2 * p + sa, True),
                                         (2 * q + sb, True),
                                         (2 * s2 + sb, False),
                                         (2 * r + sa, False)]
                                if len({m_[0] for m_ in modes}) < 4:
                                    ok = False
                                    break
                                left = [m_ for m_ in modes
                                        if m_[0] // 2 <= cut]
                                right = [m_ for m_ in modes
                                         if m_[0] // 2 > cut]
                                # XX': keep creation as n, annih as 1-n;
                                # Y'Y: the reverse
                                lp = tuple((so, occ) for so, occ in left)
                                rp = tuple((so, not occ) for so, occ in right)
                                ok = side_fits(lp) and side_fits(rp)
                                if not ok:
                                    break
                                cs_terms.append((abs(g) / 2, lp, rp))
                            if not ok:
                                break
                        if not ok:
                            penalty += 2 * abs(g)
    # far pairs: exact neutral decomposition + operator AM-GM
    for i in range(n):
        for j in range(i + 1, n):
            if j - i < ell:     # near enuc goes to windows
                m = mcount(i, j)
                for w in windows_of(i, j):
                    constW[w] += 1.0 / (d * (j - i)) / m
                continue
            g = eri[i, i, j, j]
            vi, vj = V[j, i, i], V[i, j, j]
            lower_const += 1.0 / (d * (j - i)) - g
            for site, linc in ((i, vi + g), (j, vj + g)):
                m = mcount(site, site)
                for w in windows_of(site, site):
                    linW[w][site - w] += linc / m
                    quadW[w][site - w] += g / 2 / m
    lower = lower_const - penalty
    base = [_window_operator(hW[w], eriW[w], linW[w], quadW[w], constW[w])
            for w in range(nw)]
    # balanced-eps outer loop: every eps is a valid committed inequality,
    # so the loop is pure quality. Absorptions are diagonal, so rebuilds
    # are base + diagonal vector. Optimal eps* = sqrt(<R>/<L>) turns the
    # cost into the true CS expectation bound 2 c sqrt(<L><R>).
    dim, nq = 4 ** ell, 2 * ell
    ndiag = [((np.arange(dim) >> (nq - 1 - q)) & 1).astype(float)
             for q in range(nq)]

    def side_data(modes):
        atoms = [m // 2 for m, _ in modes]
        lo, hi = min(atoms), max(atoms)
        m = mcount(lo, hi)
        out = []
        for w in windows_of(lo, hi):
            dv = np.ones(dim)
            for so, occ in modes:
                dv = dv * (ndiag[so - 2 * w] if occ
                           else 1.0 - ndiag[so - 2 * w])
            out.append((w, 1.0 / m, dv))
        return out

    sides = [(side_data(L), side_data(R)) for _, L, R in cs_terms]
    eps = np.ones(len(cs_terms))

    def assemble(eps_v):
        acc = [np.zeros(dim) for _ in range(nw)]
        for k, (c, _, _) in enumerate(cs_terms):
            for w, wt, dv in sides[k][0]:
                acc[w] -= c * eps_v[k] * wt * dv
            for w, wt, dv in sides[k][1]:
                acc[w] -= c / eps_v[k] * wt * dv
        return [base[w] + np.diag(acc[w]) for w in range(nw)]

    # the naive balance update eps* = sqrt(<R>/<L>) is NOT an ascent step
    # (it optimizes against the current minimizers; lambda_min then
    # re-minimizes and can drop — measured). Damped updates plus
    # best-by-measured-total tracking make the loop monotone-safe: the
    # eps=1 start is included, so it can never end worse than it began.
    best_eps, best_tot = eps.copy(), -math.inf
    for _ in range(cs_rounds):
        mats = assemble(eps)
        tot, p2 = 0.0, []
        for M in mats:
            lam0, v = _ground_vec(M)
            tot += lam0
            p2.append(v ** 2)
        if tot > best_tot:
            best_tot, best_eps = tot, eps.copy()
        for k in range(len(cs_terms)):
            eL = sum(wt * float(p2[w] @ dv) for w, wt, dv in sides[k][0])
            eR = sum(wt * float(p2[w] @ dv) for w, wt, dv in sides[k][1])
            bal = math.sqrt((eR + 1e-9) / (eL + 1e-9))
            eps[k] = min(20.0, max(0.05, eps[k] ** 0.7 * bal ** 0.3))
    if cs_rounds:
        mats = assemble(eps)
        tot = sum(np.linalg.eigvalsh(M)[0] for M in mats)
        if tot < best_tot:
            eps, mats = best_eps, assemble(best_eps)
    else:
        mats = assemble(eps)
    if correction_iters and nw > 1:
        D = 4 ** (ell - 1)
        Cs = _window_multipliers(mats, D, correction_iters)
        IE = np.eye(4 ** ell // D)
        for w in range(nw):
            M = mats[w]
            if w > 0:
                M = M + np.kron(Cs[w - 1], IE)
            if w < nw - 1:
                M = M - np.kron(IE, Cs[w])
            mats[w] = M
    for M in mats:
        c = eigen_bracket(M)
        lower += c.value - c.err

    # upper: product of exactly solved blocks, cross energy by exact
    # factorization of block-diagonal spin-orbital 1-RDMs
    from scipy.sparse.linalg import eigsh
    sizes = [ell] * (n // ell)
    if n % ell:
        sizes.append(n % ell)
    blocks, start = [], 0
    for size in sizes:
        blocks.append(list(range(start, start + size)))
        start += size
    blk = {i: bi for bi, b in enumerate(blocks) for i in b}
    upper = 0.0
    gammas = {}
    for bi, b in enumerate(blocks):
        idx = np.ix_(b, b)
        h_own = T[idx] + sum(V[c][idx] for c in b)
        eri_b = eri[np.ix_(b, b, b, b)]
        enuc_b = sum(1.0 / (d * (jj - ii)) for ii in b for jj in b if jj > ii)
        Hb = _fock_hamiltonian(np.eye(len(b)), h_own, eri_b, enuc_b)
        _, Vec = eigsh(Hb, k=1, which="SA")
        v = Vec[:, 0] / np.linalg.norm(Vec[:, 0])
        e = float(v @ (Hb @ v))
        upper += e + 8 * (len(Hb) + 2) * np.finfo(float).eps \
            * (float(np.linalg.norm(Hb, 1)) + abs(e))
        ann = _jw_ann(2 * len(b))
        gam = np.zeros((2 * len(b), 2 * len(b)))
        for P in range(2 * len(b)):
            for R in range(2 * len(b)):
                gam[P, R] = float(v @ (ann[P].T @ (ann[R] @ v)))
        gammas[bi] = gam
    # cross one-electron: block density in the field of other nuclei,
    # plus kinetic/attraction cross matrix elements vanish (<a'a> = 0)
    for bi, b in enumerate(blocks):
        gam = gammas[bi]
        for c in range(n):
            if blk[c] == bi:
                continue
            for ii, p in enumerate(b):
                for jj, r in enumerate(b):
                    upper += V[c, p, r] * (gam[2 * ii, 2 * jj]
                                           + gam[2 * ii + 1, 2 * jj + 1])
        for other in range(bi + 1, len(blocks)):
            upper += sum(1.0 / (d * abs(jj - ii))
                         for ii in b for jj in blocks[other])
    # cross two-electron by exact product-state factorization
    for p in range(n):
        for q in range(n):
            if blk[p] == blk[q]:
                continue
            for r in range(n):
                for s2 in range(n):
                    g = eri[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    if blk[r] == blk[p] and blk[s2] == blk[q]:
                        gA, gB = gammas[blk[p]], gammas[blk[q]]
                        ip, ir = blocks[blk[p]].index(p), \
                            blocks[blk[p]].index(r)
                        iq, is2 = blocks[blk[q]].index(q), \
                            blocks[blk[q]].index(s2)
                        dA = gA[2 * ip, 2 * ir] + gA[2 * ip + 1, 2 * ir + 1]
                        dB = gB[2 * iq, 2 * is2] + gB[2 * iq + 1, 2 * is2 + 1]
                        upper += 0.5 * g * dA * dB
                    if blk[s2] == blk[p] and blk[r] == blk[q]:
                        gA, gB = gammas[blk[p]], gammas[blk[q]]
                        ip, is2 = blocks[blk[p]].index(p), \
                            blocks[blk[p]].index(s2)
                        iq, ir = blocks[blk[q]].index(q), \
                            blocks[blk[q]].index(r)
                        for sp in range(2):
                            upper -= 0.5 * g \
                                * gA[2 * ip + sp, 2 * is2 + sp] \
                                * gB[2 * iq + sp, 2 * ir + sp]
    return Certified(0.5 * (upper + lower), 0.5 * (upper - lower),
                     Tier.RIGOROUS,
                     (f"h-chain marginal-lower ell={ell} n={n} d={d:g} "
                      f"iters={correction_iters} block-product-upper",))
