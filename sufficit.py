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
        self.dtype, scale = np.float64, None
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
            res = _compress_certified(K, tol, n_probes, scale, rng)
            if res is None:
                raise ValueError(f"block {K.shape} not compressible to "
                                 f"{tol:.3g}: full-rank probe failed")
            U, V, beta, rounds, fl = res
            self.fail_p += rounds * 10.0 ** (-n_probes)
            flops += fl
            if V.shape[0] * (K.shape[0] + K.shape[1]) < K.size:
                self.far.append((T.idx, S.idx, U, V, beta))
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
        out = np.zeros(self.n_tgt, self.dtype)
        bound, flops = np.zeros(self.n_tgt), 0
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


def _hlt_solve(N, omega, sigma, ridge=None):
    """Solve for g_t (t = 2..N) minimizing the e^(2w)-weighted L2 kernel
    deviation, then certify c = sup_w |deviation| e^w rigorously. With no
    ridge given, scans a ladder and keeps the best-certifying g — the
    bound is a posteriori, so the scan cannot compromise validity."""
    if ridge is None:
        best = min((_hlt_solve(N, omega, sigma, r)
                    for r in (1e-6, 1e-8, 1e-10, 1e-12)), key=lambda gc: gc[1])
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
