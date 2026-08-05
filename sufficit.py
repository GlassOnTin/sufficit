"""Sufficit Phase 0: the error-algebra IR, minimal seed.

A Certified value = (value, err, tier, provenance, fail_p). err bounds the
distance to the true value (abs for scalars, 2-norm for vectors) and composes
through rewrites the way derivatives compose through autodiff. Tiers degrade
to the weakest input; provenance records which rewrites produced the bound;
fail_p is the probability the bound is wrong (0 for deterministic rewrites),
accumulating through composition by union bound. A rewrite that can also
certify how hard its output leans on an input exports that as a tiered
sensitivity -- the datum composed plans use to split one error budget
across stages.

The numpy-based rewrites carry exact-arithmetic bounds (declared in their
docstrings); the scalar Phase 2 pipelines carry floating-point rounding
too, via the directed-rounding Interval type ("+fp" in provenance).
"""
from __future__ import annotations

import functools
import heapq
import math
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Callable, Tuple

import numpy as np


class Tier(IntEnum):
    EMPIRICAL = 0   # statistically validated
    ASYMPTOTIC = 1  # valid in a stated limit
    RIGOROUS = 2    # machine-checkable bound


@dataclass(frozen=True)
class Sensitivity:
    """How hard the output leans on one named input: a certified
    Lipschitz bound |output moves| <= bound * |input moves| (2-norm on
    vectors, matching err). This is the Jacobian-like datum that forward
    error propagation runs on. A rewrite that exports it lets a
    composed plan price how much input error it can afford. The bound
    is a claim like any other, so it carries a tier. When two
    sensitivities to the same input compose, bounds add and the
    weakest tier wins, which are the rules err and tier already
    obey."""
    bound: float
    tier: Tier
    wrt: str    # names the input; equal names mean the same input


@dataclass(frozen=True)
class Certified:
    value: Any                 # float or ndarray
    err: float                 # |value - true| (2-norm for vectors)
    tier: Tier
    provenance: Tuple[str, ...]
    fail_p: float = 0.0        # P(bound is wrong); composes by union bound
    # the planner's per-rung log: (rewrite, knob, predicted cost,
    # measured seconds, outcome). Bookkeeping about the run, not part
    # of the proof: provenance stays deterministic, timings live here,
    # and composition drops the receipt.
    receipt: Tuple = ()
    # amplification of input error, when the rewrite can certify one.
    # None is "no claim", never "no amplification".
    sensitivity: Sensitivity = None

    def _combine(self, other: "Certified", value, err, note: str) -> "Certified":
        s = None
        if (note in ("add", "sub") and self.sensitivity and other.sensitivity
                and self.sensitivity.wrt == other.sensitivity.wrt):
            # |d(a+-b)| <= |da| + |db|; a product has no global
            # Lipschitz constant, so mul drops the claim rather than
            # fake one
            s = Sensitivity(self.sensitivity.bound + other.sensitivity.bound,
                            min(self.sensitivity.tier, other.sensitivity.tier),
                            self.sensitivity.wrt)
        return Certified(value, err, min(self.tier, other.tier),
                         self.provenance + other.provenance + (note,),
                         min(1.0, self.fail_p + other.fail_p),
                         sensitivity=s)

    def through(self, inp: "Certified") -> "Certified":
        """The chain rule for certificates. This value was computed
        from an approximate input that carries its own certificate. If
        this certificate exports a sensitivity to that input, the
        input's error converts to output error at the exported rate.
        err grows by bound * inp.err. The tier is the weakest of the
        three claims involved: this bound, the input's, and the
        sensitivity's. fail_p accumulates by union bound. The caller
        vouches that inp certifies the very input the sensitivity
        names, in the norm the sensitivity is stated in. Refuses when
        no sensitivity is exported, rather than guessing."""
        if self.sensitivity is None:
            raise ValueError("no exported sensitivity: cannot convert "
                             "input error to output error")
        s = self.sensitivity
        return Certified(self.value, self.err + s.bound * inp.err,
                         min(self.tier, inp.tier, s.tier),
                         inp.provenance + self.provenance
                         + (f"through {s.wrt}: err += {s.bound:.3g} * "
                            f"{inp.err:.3g}",),
                         min(1.0, self.fail_p + inp.fail_p),
                         sensitivity=s)

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


def _bitcount(a):
    """np.bitwise_count, with a SWAR popcount fallback so the module
    runs on numpy 1.26 (bitwise_count arrived in numpy 2.0)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    x = np.asarray(a).astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * np.uint64(0x0101010101010101))
            >> np.uint64(56)).astype(np.int64)


def _dn(x):
    return math.nextafter(x, -math.inf)


def _up(x):
    return math.nextafter(x, math.inf)


class Interval:
    """Directed-rounding interval via outward nextafter widening: field ops
    widen 1 ulp (IEEE round-to-nearest is within 1/2 ulp), transcendentals
    2 ulps under the ASSUMPTION of faithful (<= 1 ulp) libm, the one
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
# schedule, the search that should discover it is future work.

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
# a certified H-matrix. Only kernel(tgt_pts, src_pts) -> block is assumed,
# no expansions, no smoothness proofs. Each admissible block is compressed
# by the randomized range finder and certified a posteriori by Gaussian
# probes (operator-norm bound, so it holds for EVERY later charge vector);
# fail_p union-bounds over all blocks and rounds. Build cost is ~N^2
# kernel evals (certification requires touching every block once), so
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
    meets tol, a wrong guess just loses the competition."""
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
    block by pair-direction phases D_T K D_S is a UNITARY transformation,
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
    never departs along a used edge, at most Delta*(Delta-1)^(n-1)
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
    """Coefficients of Psi(S), the cluster sum touching the vertex set S
    exact through total size 10: singles, plus Ursell pairs whose
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
    Ising model, via pinned clusters. Every subgraph with odd set {a,b}
    is one connected pinned polymer w0 times an even gas off its
    vertices, so <s_a s_b> = sum_w0 t^|w0| exp(-Psi(V(w0))), with Psi
    the cluster sum touching V(w0). Two errors are carried. The pinned
    tail: a pinned polymer has exactly two odd vertices, hence an
    Eulerian path a->b, so the count is <= 4*3^(n-1) and the dressing is
    bounded by e^((n+1)B). And the Psi truncation at size 8 (tilted KP,
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
    best-certifying g. The bound is a posteriori, so the scan cannot
    compromise validity. Cached, so callers treat g as read-only."""
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
    fail_p = 2*erfc(z/sqrt(2)).

    The certificate also exports its sensitivity. The value is the
    linear map g.C, so a correlator error delta moves the value by at
    most |g|*|delta|. It also moves the kernel-mismatch bill, which is
    anchored at the true C(1), by up to c*delta(1). Cauchy-Schwarz folds
    both into the one constant sqrt(c^2 + |g|^2). That is an exact norm,
    hence RIGOROUS whatever the tier of the value's own bound, and it is
    valid because both densities are nonnegative (already assumed). This
    is the datum a composed plan needs to decide how much correlator
    error it can afford."""
    N = len(C)
    g, c = _hlt_solve(N, omega, sigma)
    value = float(g @ C[1:])
    sens = Sensitivity(_up(math.hypot(c, float(np.linalg.norm(g)))),
                       Tier.RIGOROUS, "correlator")
    if cov is None:
        return Certified(value, c * float(C[0]), Tier.RIGOROUS,
                         (f"hlt-smeared omega={omega:g} sigma={sigma:g} "
                          f"c={c:.3g} assumes rho>=0",),
                         sensitivity=sens)
    stat = z * math.sqrt(float(g @ cov[1:, 1:] @ g))
    err = c * (float(C[0]) + z * math.sqrt(float(cov[0, 0]))) + stat
    return Certified(value, err, Tier.EMPIRICAL,
                     (f"hlt-smeared omega={omega:g} sigma={sigma:g} "
                      f"c={c:.3g} z={z:g} assumes rho>=0",),
                     fail_p=2 * math.erfc(z / math.sqrt(2)),
                     sensitivity=sens)


# ------------------------------------------------------------- Phase 4:
# Mori-Zwanzig closures. Rigorous tier: linear slow-fast systems, where
# the memory kernel K(s) = A12 e^(A22 s) A21 decays at the fast sector's
# spectral gap and the Markovian closure carries a Gronwall bound with
# computable constants (log-norms, block norms), the bound's provenance
# is the gap, and without one the rewrite refuses. Empirical tier:
# distribution-free conformal calibration for closures of ANY system
# (nonlinear included), with rigorous failure probability 1/(n_cal+1)
# under exchangeability of initial conditions, the honest certificate
# this phase exists to ship. Both exact-arithmetic.


def _lognorm(M):
    """2-norm logarithmic norm: ||e^(Mt)|| <= e^(lognorm(M) t)."""
    return float(np.linalg.eigvalsh((M + M.T) / 2)[-1])


def mz_closure_linear(A: np.ndarray, k: int, x0: np.ndarray,
                      T: float) -> Certified:
    """Phase 4 rewrite, rigorous tier: x_slow(T) for dx/dt = Ax with the
    first k coordinates slow observables (x0 is the FULL initial state),
    via the Markovian closure Ar = A11 - A12 A22^{-1} A21. The error
    bound is ||K(s)|| <= kappa e^(-mu s), with mu the fast-sector gap
    (-lognorm of A22), carried by Gronwall through the reduced
    propagator. A nonzero fast initial condition contributes its
    decaying transient kappa12 ||x20|| G2. Refuses when mu <= 0."""
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
    exchangeability, a distribution-free guarantee, no model
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
    closure error itself, "slow variables" are the split the machine
    certifies tightest. Starts from targets (coordinates the caller must
    keep); with none, seeds from the best single-or-pair split (pure
    greedy is myopic: a slow coordinate left in the fast sector kills the
    gap, and the structure only shows at pair level, deeper hidden
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
# flat-spectrum oscillatory operator), the bound's physical provenance
# is the scattering strength k^2 * contrast * area. The Neumann-series depth
# comes from the FAR-FIELD tolerance (query-first); the far-field
# functional propagates by Cauchy-Schwarz. Certificates are for the
# STATED discrete system (midpoint Nystrom, equal-area-disk self term);
# continuum discretization error is the named gap (asymptotic tier
# territory). Strong scattering needs resolvent-based certification,
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
# certificate is A POSTERIORI on the assembled operator, probes of
# K w - B w, so validity never depends on the construction.


def _bisect_positions(pts, idx):
    """Positions splitting a cluster in two along its wider axis."""
    xs = pts[idx]
    key = xs.real if np.ptp(xs.real) >= np.ptp(xs.imag) else xs.imag
    order = np.argsort(key, kind="stable")
    return order[:len(idx) // 2], order[len(idx) // 2:]


def _quad_positions(pts, idx):
    """Positions splitting a cluster in four (two bisections): in 2D the
    DIAMETER must halve per butterfly level, which one binary split does
    not do, with branching 2 the stage rank products only shrink like
    2^(L/2) and every stage stays expensive (measured before this fix)."""
    a, b = _bisect_positions(pts, idx)
    a1, a2 = _bisect_positions(pts, idx[a])
    b1, b2 = _bisect_positions(pts, idx[b])
    return [a[a1], a[a2], b[b1], b[b2]]


class ButterflyBlock:
    """Butterfly factorization of one oscillatory kernel block. The
    guarantee per apply(q) is ||value - K q||_2 <= beta ||q||_2 with the
    stated fail_p. beta is certified a posteriori by probes on the
    assembled factorization. Setup touches the dense block, as with the
    H-matrix, so the value is amortized applies. Tier RIGOROUS (exact
    arithmetic)."""

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
# quotient of a heuristic Lanczos vector, the variational theorem
# certifies it no matter where the vector came from. Lower bound:
# Cholesky feasibility proofs of H - cI >= 0 (the poor man's SDP dual
# certificate), bisected from a rigorous Gershgorin seed; a spurious FP
# Cholesky failure only loosens the bracket, never invalidates it.
# Exact-arithmetic tier (Cholesky success is FP-trusted, declared).
# The 2-RDM SDP lower bound now exists further down as
# rdm2_energy_bracket, priced in orbitals rather than in states. A
# molecular-integrals pipeline is still a named next rung. This
# demonstrates the bracket structure itself.


_GPU = {"on": False}


def use_gpu(on: bool = True):
    """Route fp32 Cholesky certification through CuPy (consumer-GPU fp64
    is ~1:64 throttled; fp32 with honestly widened pads is the move).
    Validity is unaffected either way, the pads carry the working eps
    and the measured casting error. Requires cupy when enabled."""
    if on:
        import cupy                          # noqa: F401 - availability
    _GPU["on"] = bool(on)


def _arpack_v0(n):
    """A fixed start vector for ARPACK, because the ground state it is
    asked for is not always unique. A Heisenberg window on an odd
    number of sites has a spin-1/2 ground DOUBLET: measured, the gap
    above the ground energy is 4e-16 at ell=5 and 7e-16 at ell=7,
    against 1e-1 at ell=6 and 6e-2 at ell=8. Every member of that
    doublet has the same energy -- ARPACK converges to within 1e-15 of
    the true value at every width -- but the block upper bound needs
    more than the energy. It reads the edge spin expectations to price
    the bond between neighbouring blocks, and the two members of a
    doublet carry opposite edge magnetization, so which one comes back
    moves the bracket.

    Left to itself, eigsh draws its start from numpy's global random
    stream, so which member came back depended on how many other eigsh
    calls had run earlier in the same process: measured, the ell=7
    window at N=10 returned 0.382, 0.332 and 0.379 on three
    consecutive calls, while the gapped ell=6 and ell=8 did not move.
    That is a variational trial vector, so an unlucky member loosens a
    bracket and cannot invalidate one -- but an unrepeatable
    certificate is the worse failure here. Seeded PCG64: generic
    overlap, same answer every run, and the same stream on every numpy
    since 1.17. Which member a given LAPACK picks from the degenerate
    subspace is still its own business, so ell=5 and ell=7 can differ
    between scipy versions where the gapped widths agree exactly.
    Choosing the member that minimizes the upper bound, rather than
    accepting the one handed over, is an available tightening that
    nobody has cashed in."""
    return np.random.default_rng(0).standard_normal(n)


def eigen_bracket(H: np.ndarray, tol: float = None,
                  fp32: bool = False) -> Certified:
    """Certified bracket on lambda_min(H), H Hermitian: value +- err
    contains the true ground energy. Raises if tol is given and the
    achieved width exceeds it. fp32=True runs the Cholesky feasibility
    proofs in float32, optionally on GPU via use_gpu. Rigor is kept
    honest there by two additional carried terms. One is the MEASURED
    casting error ||H - fl32(H)||_F, computed exactly in f64, a
    posteriori, so nothing is assumed. The other is Higham-style margins
    at fp32 eps. The pads land at about 1e-4 scale, which is negligible
    against mHa brackets."""
    H = np.asarray(H)
    n = len(H)
    if n < 64:                              # dense: exact vector, and no
        _, V = np.linalg.eigh(H)            # Lanczos k >= N warnings on
        v = V[:, 0]                         # tiny sector matrices
    else:
        from scipy.sparse.linalg import eigsh
        try:
            _, V = eigsh(H, k=1, which="SA", v0=_arpack_v0(n))
            v = V[:, 0]
        except Exception:                   # heuristic source; bracket
            v = np.ones(n)                  # stays valid, just looser
    v = v / np.linalg.norm(v)
    up = float(np.real(v.conj() @ (H @ v)))     # variational theorem

    if fp32:
        H32 = H.astype(np.float32)
        cast_err = float(np.linalg.norm(H - H32.astype(np.float64)))
        eps_w = float(np.finfo(np.float32).eps)
        if _GPU["on"]:
            import cupy as xp
            Hw_, eye_w = xp.asarray(H32), xp.eye(n, dtype=xp.float32)
        else:
            xp, Hw_, eye_w = np, H32, np.eye(n, dtype=np.float32)

        def psd(c):
            # CuPy's non-PSD behaviour varies by version: exception OR
            # NaNs. Belt and braces, any doubt counts as failure, which
            # only loosens the bound (the safe direction).
            try:
                L = xp.linalg.cholesky(Hw_ - np.float32(c) * eye_w)
                return bool(xp.isfinite(L).all())
            except Exception:
                return False
    else:
        cast_err, eps_w = 0.0, float(np.finfo(np.float64).eps)

        def psd(c):
            try:
                np.linalg.cholesky(H - c * np.eye(n))
                return True
            except np.linalg.LinAlgError:
                return False

    d = np.real(np.diag(H))
    gersh = float(np.min(d - (np.sum(np.abs(H), axis=1) - np.abs(d))))
    r = float(np.linalg.norm(H @ v - up * v))
    lo = up - 2 * r - max(1e-12, 4 * eps_w) * (1 + abs(up))
    if not psd(lo):
        # Gershgorin IS a certificate by itself, no Cholesky proof
        # needed (and for exactly-degenerate cases H - gersh*I is
        # singular, so demanding one would wrongly fail)
        lo = gersh
    hi = up
    floor = max(1e-13, 0.25 * eps_w) * (1 + abs(up))
    for _ in range(60):
        if hi - lo <= floor:
            break
        c = 0.5 * (lo + hi)
        if psd(c):
            lo = c
        else:
            hi = c
    # carry the FP margins: the Rayleigh quotient's evaluation error
    # (f64) and Cholesky's backward-stability slack at the WORKING eps
    # (success proves PSD of H - cI + E, ||E|| <~ n eps ||H||, Higham),
    # plus the measured fp32 casting error when applicable
    norm1 = float(np.linalg.norm(H, 1))
    up = up + 8 * (n + 2) * np.finfo(np.float64).eps * (norm1 + abs(up))
    lo = lo - 8 * (n + 2) * eps_w * (norm1 + abs(up)) - cast_err
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
# Cholesky afterwards, the optimizer is pure quality. Richer correction
# structures than single-overlap operators remain (the fully general
# dual); this family already closes much of the gap. The
# upper half: a product of per-block Lanczos states, block energies and
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
    constraints, linear objective, symmetrize any feasible point);
    non-consecutive overlap variables are redundant (their consistency is
    implied by consecutive pairs); and decomposition-weight freedom is
    absorbed (verified numerically: joint (C, weights) ascent reaches the
    identical plateau to six digits). The residual gap at fixed ell is
    the relaxation level itself, the hierarchy knob is ell.
    Bundle: lambda_min(W(C)) = min_v of exact affine cuts
    <v|Hw|v> + <rhoL(v) - rhoR(v), C>; master dual is a simplex QP over
    cut weights, solved by exponentiated gradient. iters = oracle calls
    (eigendecompositions); certification happens downstream, so this is
    pure quality. Ten is the default because ten is where the quality
    stops arriving. Measured on two numpy stacks at N = 10, 40 and 60
    across ell = 4..9, ten calls buy 98-107% of the bracket tightening
    that eighty buy, and the bracket they return sits within 1.1% of
    the eighty-call bracket in either direction. Above 100% means
    eighty is the WORSE bracket, which is not a contradiction: the
    ascent optimizes the dual of the uniform-weight window, while the
    bracket applies the resulting C to the weighted sliding windows,
    so a better dual value on the reference window is not obliged to
    give a tighter bracket. At ell = 3 the ascent moves nothing at
    all, at any N. Eighty calls cost 20-36x more than ten rather than
    8x, because the bundle grows toward its cut cap and the master QP
    grows with it. These numbers only became measurable once the
    ARPACK start vector was pinned -- see _arpack_v0."""
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


@functools.lru_cache(maxsize=None)
def _window_lower(N, width, iters):
    """Certified lower bound on the N-site chain from sliding windows
    of the given width: weighted windows summing exactly to H, plus
    exactly telescoping corrections (+C on a window's left overlap, -C
    on its neighbour's right overlap).

    The width need not be the rung's ell. Any width bounds the chain,
    and a narrower one is cheaper, so ell caps the width rather than
    fixing it -- which matters because this relaxation has a PARITY
    structure and a wider window is not always a better one. Measured
    at N=40 with corrections off: -19.616 at width 3 against -21.169 at
    width 4, -19.044 at 5 against -19.710 at 6, -18.766 at 7 against
    -19.136 at 8. Every even width is beaten by the odd width below it.
    The reason is not numerical: an even-site open segment can close
    into a complete singlet covering, so its ground energy per bond
    sits lower, and a bound that sums window minima is looser the lower
    those minima go. The multiplier ascent knows it and works harder on
    even widths -- it buys 1.49 at width 4 against 0.56 at width 5 --
    without ever closing the gap.

    Memoized on (N, width, iters) because the planner walks the ladder
    and each width would otherwise be recomputed at every rung above
    it."""
    C = _chain_correction(width, iters) if iters else None
    m = [min(N - width, i) - max(0, i - width + 2) + 1 for i in range(N - 1)]
    cache, lower = {}, 0.0
    for w in range(N - width + 1):
        key = (tuple(1.0 / m[w + j] for j in range(width - 1)),
               w >= 1, w <= N - width - 1)
        if key not in cache:
            Hw = _heis_window(key[0])
            if C is not None and key[1]:
                Hw = Hw + np.kron(C, np.eye(2))
            if C is not None and key[2]:
                Hw = Hw - np.kron(np.eye(2), C)
            c = eigen_bracket(Hw)
            cache[key] = c.value - c.err
        lower += cache[key]
    return lower


@functools.lru_cache(maxsize=None)
def _heis_block(size):
    """Ground state of an isolated size-site Heisenberg segment, with
    the two edge spin expectations the product bound joins blocks by.
    It depends on nothing but the size, and the tiling search asks for
    the same handful of sizes over and over -- across candidate cuts,
    across window widths, across chain lengths -- so it is memoized
    once for the process rather than recomputed per bracket."""
    from scipy.sparse.linalg import eigsh
    Hb = _heis_window((1.0,) * (size - 1))
    _, V = eigsh(Hb, k=1, which="SA", v0=_arpack_v0(Hb.shape[0]))
    v = V[:, 0] / np.linalg.norm(V[:, 0])
    e = float(v @ (Hb @ v))
    pad = 8 * (2 ** size + 2) * np.finfo(float).eps \
        * (float(np.linalg.norm(Hb, 1)) + abs(e))
    return (e + pad, _site_spin_expect(v, size, 0),
            _site_spin_expect(v, size, size - 1))


def _block_tilings(N, ell, merge_lone=True):
    """Ways to cut N sites into blocks for the product upper bound.

    Any cut gives a valid bound -- the bound is the energy of a
    product state, and the variational theorem does not care how the
    factors were chosen -- so this is a free choice, and the obvious
    one is not always the best. Cutting greedily into blocks of ell
    leaves a remainder, and a runt block is a poor bargain: at N=40,
    ell=9 the greedy [9,9,9,9,4] bounds 0.22 worse than an even
    [8,8,8,8,8]. But bigger blocks capture more correlation, so the
    runt is sometimes worth its keep: at N=60, ell=8 the greedy
    [8]*7+[4] beats the even split by 0.26, the other way. No rule
    wins both, which is why this returns candidates rather than an
    answer -- the caller takes the lowest, and the minimum of valid
    upper bounds is a valid upper bound.

    The narrower windows' cuts are candidates too, and that is not
    generosity: it is what stops a wider window from bounding worse
    than a narrower one. At N=10 the ell=6 cut (6,4) beat everything
    ell=7 generated for itself, so the ladder ran backwards at a rung
    where nothing physical had gone wrong -- the wider window simply
    was not offered the narrower one's cut, though it can afford it.
    Offering every cut up to ell makes the upper bound non-increasing
    in ell by construction. Cost stays inside the rung: a block of
    size s costs 2^s and no candidate is wider than the greedy cut's
    widest."""
    def greedy(width):
        g = [width] * (N // width)
        if N % width == 1 and merge_lone:
            g[-1] += 1          # a lone site is worse than one long block
        elif N % width:
            g.append(N % width)
        return tuple(g)

    # merge_lone widens one block past ell to swallow a lone remainder,
    # which the spin chain has always done and which costs it a factor
    # of two. On the hydrogen chain the same block costs a factor of
    # four, and it is not worth it: the search below already offers
    # cuts that beat a lone atom without going over budget.

    out = [greedy(w) for w in range(2, ell + 1)]
    wide = max(out[-1])
    for k in (-(-N // ell), -(-N // ell) + 1):
        q, r = divmod(N, k)
        if q >= 2 and (q + 1 if r else q) <= wide:
            out.append(tuple([q + 1] * r + [q] * (k - r)))
    return list(dict.fromkeys(out))


def heisenberg_chain_bracket(N: int, ell: int = 8,
                             correction_iters: int = 10) -> Certified:
    """Certified two-sided bracket on the ground energy of the spin-1/2
    Heisenberg open chain of N sites, at cost 2^ell independent of N.
    correction_iters=0 disables the SDP-dual multiplier ascent.

    The upper bound is the energy of a product of per-block ground
    states, so every term in it is an explicit Rayleigh quotient. The
    blocks meet at junctions, and a junction costs <S>.<S> between the
    two edge sites facing each other. A block of odd width has a
    degenerate ground doublet, and its two members carry opposite edge
    magnetization, so that junction term arrives with an arbitrary
    sign -- ARPACK's, in effect. It need not: a global spin flip is a
    symmetry of the isotropic Hamiltonian, so the flipped block is an
    equally exact ground state at exactly the same energy with both
    edge components negated. Flipping whenever a junction would be
    positive makes every junction help. Greedy left to right is the
    best any orientation can do, because the junctions form an open
    chain with no field, so each choice is free given the one before.
    Measured gain, per bond, at N=200: 0.0726 at ell=3, 0.0257 at
    ell=5, 0.0120 at ell=7 -- and exactly zero at every even width,
    whose singlet blocks have no edge magnetization to orient."""
    if N <= ell:
        c = eigen_bracket(_heis_window((1.0,) * (N - 1)))
        return replace(c, provenance=(f"chain-bracket exact N={N}",))
    # lower: the best relaxation the rung can afford. A width narrower
    # than ell bounds the chain just as validly and is cheaper, so ell
    # caps the width rather than fixing it, and the maximum of valid
    # lower bounds is a valid lower bound.
    lower = max(_window_lower(N, w, correction_iters)
                for w in range(2, ell + 1))
    # upper: product of per-block Lanczos states; all terms are explicit
    # Rayleigh quotients of the product state
    def product_upper(sizes):
        upper, prev_edge = 0.0, None
        for size in sizes:
            e, left, right = _heis_block(size)
            upper += e
            if prev_edge is not None:     # cross-block bond <S>.<S>
                bond = prev_edge[0] * left[0] + prev_edge[1] * left[1]
                if bond > 0:              # flip this block; edges negate
                    bond, right = -bond, [-x for x in right]
                upper += bond
            prev_edge = right
        return upper

    upper = min(product_upper(s) for s in _block_tilings(N, ell))
    return Certified(0.5 * (upper + lower), 0.5 * (upper - lower),
                     Tier.RIGOROUS,
                     (f"chain-bracket marginal-lower ell={ell} N={N} "
                      f"iters={correction_iters} product-upper",))


# ------------------------------------------- molecular-integrals pipeline:
# the energy bracket on an actual molecule. H2 in STO-3G needs only
# s-type Gaussian integrals, which have complete closed forms via the
# Boys function, fully self-contained (no quantum-chemistry package;
# the only external data is the published STO-3G hydrogen basis,
# Hehre-Stewart-Pople). Second quantization via Jordan-Wigner on 4 spin
# orbitals gives a 16x16 Fock-space Hamiltonian, bracketing it certifies
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
    (hartree) at bond length R bohr, over all particle-number
    sectors."""
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
# lies strictly below the s-only bracket, a theorem about two
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
    (Lowdin handles any overlap, the S11 != 1 lesson)."""
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


def _fock_hamiltonian(S, h, eri, enuc, dense=True):
    """Lowdin-orthogonalize, Jordan-Wigner, assemble the 4^nao Fock-space
    Hamiltonian (sparse internally, dense out)."""
    from scipy import sparse
    w, U = np.linalg.eigh(S)
    Xo = U @ np.diag(w ** -0.5) @ U.T
    h = Xo @ h @ Xo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, Xo, Xo, Xo, Xo)
    nso = 2 * len(h)
    n_sp = len(h)
    terms = []
    for p in range(n_sp):
        for q in range(n_sp):
            if h[p, q] != 0.0:
                for sp in range(2):
                    terms.append((h[p, q],
                                  [(2 * p + sp, True), (2 * q + sp, False)]))
    for p in range(n_sp):
        for q in range(n_sp):
            for r in range(n_sp):
                for s2 in range(n_sp):
                    g = eri[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    for sa in range(2):
                        for sb in range(2):
                            terms.append((0.5 * g,
                                          [(2 * p + sa, True),
                                           (2 * q + sb, True),
                                           (2 * s2 + sb, False),
                                           (2 * r + sa, False)]))
    H = _fermion_assemble(nso, terms) \
        + sparse.identity(2 ** nso, format="csr") * enuc
    return np.asarray(H.todense()) if dense else H.tocsr()


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
# molecular integrals x marginal-SDP bracket, on hydrogen chains, a
# certified two-sided bracket for molecular Fock spaces too big to form.
# Terms are classified by orbital spread (JW strings stay inside
# contiguous windows, so window lambda_min bounds are valid for
# fermions). The genuinely molecular difficulty is long-range Coulomb,
# is handled exactly per far atom pair: F_ij = g(n_i-1)(n_j-1)
# + (v+g)n_i + (v+g)n_j + (1/R-g), with the quadratic part bounded by
# operator AM-GM, g(n_i-1)(n_j-1) >= -(g/2)[(n_i-1)^2 + (n_j-1)^2],
# local charge-fluctuation penalties absorbed into windows; linear parts
# and constants exact. Remaining far terms use CAUCHY-SCHWARZ ABSORPTION
# (g XY + h.c. >= -|g|(XX' + Y'Y)): the JW Z-strings are unitary and
# vanish inside XX', which for distinct modes is a pure occupation
# product (creation -> n, annihilation -> 1-n; reversed for Y'Y), so
# every far term absorbs into diagonal occupation products on its two
# compact sides, local to windows, with small ground-state expectations
# replacing the flat norms that used to dominate the gap (1.04 of
# 1.79 Ha at H6/ell=3; the absorption recovered ~0.52). Repeated-mode
# or window-overflowing sides fall back to flat bounds (rare, tiny).
# Shared-C window multipliers below add a further modest tightening.
# The balanced-eps outer loop (cs_rounds) is monotone-safe by
# best-tracking; measured on H-chains its optimum is eps=1, the
# balancing family is exhausted at the start here, and the naive
# greedy update would LOSE 5 mHa/atom (not an ascent step). Upper: product of exactly solved atom blocks, cross energies
# by exact factorization of block-diagonal 1-RDMs (fermionic signs are
# benign, cross operators move in even pairs). No correction
# multipliers yet (the Heisenberg bundle machinery is the named
# tightening path).


@functools.lru_cache(maxsize=None)
def _h_chain_basis(n, d):
    """Lowdin-orthogonalized (T, V[c], eri, enuc) for the n-atom hydrogen
    chain at spacing d bohr, STO-3G. V[c] is the attraction to nucleus c
    separately, the far-pair decomposition needs it."""
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


def _fermion_assemble(nq, terms):
    """Sparse operator from ladder-operator strings [(coef, [(mode,
    dagger), ...])], vectorized over the occupation basis with bit
    arithmetic: each string is a signed partial permutation (one nonzero
    per column), validity by bit tests, JW string parity by
    bitwise_count, target state by XOR. No matrix products. Chunks are
    tree-merged to keep csr additions O(nnz log #chunks)."""
    from scipy import sparse
    dim = 1 << nq
    ks = np.arange(dim, dtype=np.int64)
    chunks, rows, cols, vals, pending = [], [], [], [], 0
    for coef, ops in terms:
        cur, ok = ks, np.ones(dim, bool)
        sign = np.ones(dim)
        for p, dag in reversed(ops):        # rightmost operator first
            bitpos = nq - 1 - p
            bit = (cur >> bitpos) & 1
            ok = ok & (bit == (0 if dag else 1))
            if p > 0:
                mask = ((1 << p) - 1) << (nq - p)
                sign = sign * (1.0 - 2.0 * (_bitcount(
                    (cur & mask).astype(np.uint64)).astype(np.int64) & 1))
            cur = cur ^ (1 << bitpos)
        idx = np.nonzero(ok)[0]
        if len(idx):
            rows.append(cur[idx])
            cols.append(idx)
            vals.append(coef * sign[idx])
            pending += len(idx)
        if pending > 64 * dim:
            chunks.append(sparse.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(dim, dim)).tocsr())
            rows, cols, vals, pending = [], [], [], 0
    if rows:
        chunks.append(sparse.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(dim, dim)).tocsr())
    if not chunks:
        return sparse.csr_matrix((dim, dim))
    while len(chunks) > 1:                  # balanced tree merge
        chunks = [chunks[i] + chunks[i + 1] if i + 1 < len(chunks)
                  else chunks[i] for i in range(0, len(chunks), 2)]
    return chunks[0]


def _occ_diag(nq, q):
    """Occupation diagonal of qubit q as a vector."""
    return ((np.arange(1 << nq) >> (nq - 1 - q)) & 1).astype(float)


def _window_operator(hw, eriw, lin, quad, const, extras=()):
    """Sparse-assembled window operator on 2*len(hw) spin orbitals:
    one- and two-electron parts, per-site linear n and -(quad)(n-1)^2
    charge penalties, a constant, and diagonal occupation-product extras
    [(coef, ((local_spin_orbital, occupied_bool), ...)), ...] from the
    Cauchy-Schwarz far-term absorption."""
    from scipy import sparse
    nsp = len(hw)
    nq = 2 * nsp
    dim = 4 ** nsp
    terms = []
    for p in range(nsp):
        for q in range(nsp):
            if hw[p, q] != 0.0:
                for sp in range(2):
                    terms.append((hw[p, q],
                                  [(2 * p + sp, True), (2 * q + sp, False)]))
    for p in range(nsp):
        for q in range(nsp):
            for r in range(nsp):
                for s2 in range(nsp):
                    g = eriw[p, r, q, s2]
                    if abs(g) < 1e-14:
                        continue
                    for sa in range(2):
                        for sb in range(2):
                            terms.append((0.5 * g,
                                          [(2 * p + sa, True),
                                           (2 * q + sb, True),
                                           (2 * s2 + sb, False),
                                           (2 * r + sa, False)]))
    H = _fermion_assemble(nq, terms)
    diag = np.full(dim, float(const))
    for i in range(nsp):
        num = _occ_diag(nq, 2 * i) + _occ_diag(nq, 2 * i + 1)
        if lin[i] != 0.0:
            diag += lin[i] * num
        if quad[i] != 0.0:
            diag -= quad[i] * (num - 1.0) ** 2
    for coef, pattern in extras:
        dv = np.full(dim, coef)
        for so, occ in pattern:
            od = _occ_diag(nq, so)
            dv = dv * (od if occ else 1.0 - od)
        diag += dv
    return (H + sparse.diags(diag)).tocsr()


def _ground_vec(M):
    """Ground eigenpair; Lanczos above small dims (only the vector is
    needed by the multiplier oracle and the eps loop)."""
    if M.shape[0] < 512:
        lam, Vv = np.linalg.eigh(np.asarray(M.todense())
                                 if not isinstance(M, np.ndarray) else M)
        return lam[0], Vv[:, 0]
    from scipy.sparse.linalg import eigsh
    lam, Vv = eigsh(M, k=1, which="SA", v0=_arpack_v0(M.shape[0]))
    return float(lam[0]), Vv[:, 0]


@functools.lru_cache(maxsize=None)
def _sector_indices(nsp):
    """Occupation sectors (N_up, N_down) of the 4^nsp window basis (JW
    qubit q <-> bit nq-1-q; even qubits spin-up). Every window term
    conserves both, so window operators are block-diagonal here, the
    certification wall at large ell is dissolved by symmetry, not by
    sparse factorization: the largest ell=7 sector is ~1225-dim."""
    nq, dim = 2 * nsp, 4 ** nsp
    ks = np.arange(dim)
    nup = sum((ks >> (nq - 1 - q)) & 1 for q in range(0, nq, 2))
    ndn = sum((ks >> (nq - 1 - q)) & 1 for q in range(1, nq, 2))
    sectors = {}
    for k in range(dim):
        sectors.setdefault((int(nup[k]), int(ndn[k])), []).append(k)
    return tuple(np.array(ix) for ix in sectors.values())


def _sector_mask(nsp):
    """Boolean mask of sector-conserving entries for a 4^nsp operator."""
    nq, dim = 2 * nsp, 4 ** nsp
    ks = np.arange(dim)
    nup = sum((ks >> (nq - 1 - q)) & 1 for q in range(0, nq, 2))
    ndn = sum((ks >> (nq - 1 - q)) & 1 for q in range(1, nq, 2))
    key = nup * (nsp + 1) + ndn
    return key[:, None] == key[None, :]


def _eigen_bracket_sectored(M, fp32=False):
    """(lower, upper) bracket on lambda_min of a sparse symmetric window
    operator via its occupation sectors: dense eigen_bracket per sector,
    plus a rigorous additive penalty for any off-sector coupling
    (Frobenius bound; ~0 by construction, never assumed)."""
    nsp = round(math.log(M.shape[0]) / math.log(4))
    total_f2 = float((M.multiply(M)).sum())
    lows, ups, diag_f2 = [], [], 0.0
    for idx in _sector_indices(nsp):
        sub = np.asarray(M[np.ix_(idx, idx)].todense())
        diag_f2 += float(np.sum(sub * sub))
        c = eigen_bracket(sub, fp32=fp32)
        lows.append(c.value - c.err)
        ups.append(c.value + c.err)
    off = math.sqrt(max(0.0, total_f2 - diag_f2))
    return min(lows) - off, min(ups) + off


def _sectored_ground(M):
    """(lambda_min, full ground vector) of a sector-block-diagonal sparse
    window operator, via dense selected-eigenpair solves per occupation
    sector (largest ell=7 sector ~1225-dim), replaces full-space
    Lanczos on 4^ell."""
    from scipy.linalg import eigh as dense_eigh
    nsp = round(math.log(M.shape[0]) / math.log(4))
    best = (math.inf, None, None)
    for ix in _sector_indices(nsp):
        sub = np.asarray(M[np.ix_(ix, ix)].todense())
        lam, vec = dense_eigh(sub, subset_by_index=[0, 0])
        if lam[0] < best[0]:
            best = (float(lam[0]), ix, vec[:, 0])
    v = np.zeros(M.shape[0])
    v[best[1]] = best[2]
    return best[0], v


def _window_multipliers(mats, D, iters):
    """Proximal-bundle ascent over PER-OVERLAP Hermitian corrections
    C_1..C_{nw-1} (see history). Scaled for large D (ell=6: D=1024,
    millions of parameters): window operators stay sparse with
    kron-structured matvecs (kron(C,I)x = (C X).ravel, never
    materialized), the Lanczos oracle is warm-started and loose-tol
    (it only guides the optimizer; certification is separate), and cuts
    are stored FACTORED, each overlap block is +Vl Vl' - Vr' Vr, two
    rank-E outer products, so gram entries are ||A'B||_F^2 sums. C stays
    dense per overlap (matvec-optimal). Returns [C_w]."""
    from scipy.linalg import eigh as dense_eigh
    dim = mats[0].shape[0]
    nw, E, nov = len(mats), dim // D, len(mats) - 1
    # sector-restricted oracle: C is enforced sector-conserving, so the
    # corrected windows stay block-diagonal and lambda_min = min over
    # sector sub-blocks, dense selected-eigenpair solves (<= ~1225-dim
    # at ell=7) replace full-space Lanczos, ~12x on the dominant ell=7
    # cost. Correction sub-blocks extract from the kron structure by
    # index arithmetic: kron(C,I_E)[a,b] = C[a//E,b//E] (a%E == b%E),
    # kron(I_E,C)[a,b] = C[a%D,b%D] (a//D == b//D).
    nsp_w = round(math.log(dim) / math.log(4))
    sec_ix = _sector_indices(nsp_w)
    # sector blocks extracted ON DEMAND and cached: Weyl pruning means
    # most sectors are never solved, and eager densification of all of
    # them costs ~1.3 GB/window at ell=8, the ell=8 memory wall
    _bs_cache = {}

    def base_sec(w, si):
        if (w, si) not in _bs_cache:
            ix = sec_ix[si]
            _bs_cache[(w, si)] = np.asarray(mats[w][np.ix_(ix, ix)]
                                            .todense())
        return _bs_cache[(w, si)]
    lidx = [ix // E for ix in sec_ix]
    lmask = [(ix % E)[:, None] == (ix % E)[None, :] for ix in sec_ix]
    ridx = [ix % D for ix in sec_ix]
    rmask = [(ix // D)[:, None] == (ix // D)[None, :] for ix in sec_ix]

    # Weyl-pruned branch-and-bound over sectors: each sector's cached
    # lambda plus the path-length drift ||Delta C||_F accumulated since
    # its last solve (Weyl + triangle inequality) gives a rigorous
    # optimistic bound; solve in optimistic order, stop when the next
    # bound cannot beat the best found. Exact minimum, ~10x fewer eigh
    # calls than solving every sector every oracle call (whose per-call
    # driver overhead measured 10x SLOWER than Lanczos at ell=6).
    nsec = len(sec_ix)
    # -inf = never solved: no information, optimistically arbitrarily low
    lam_cache = [[-math.inf] * nsec for _ in range(nw)]
    # per-window per-sector drift budgets from correction SUB-BLOCK
    # norms: a sector's eigenvalue shifts by at most the Weyl norm of
    # the correction change restricted to its own overlap block
    # (unmasked Frobenius upper-bounds the masked sub-block). Delta C is
    # built from ground densities in the few minimal sectors, so most
    # sectors' budgets barely grow and their cached lambdas stay valid.
    budgets = [np.zeros(nsec) for _ in range(nw)]
    lflat = [(lidx[si][:, None].astype(np.int64) * D
              + lidx[si][None, :]).ravel() for si in range(nsec)]
    rflat = [(ridx[si][:, None].astype(np.int64) * D
              + ridx[si][None, :]).ravel() for si in range(nsec)]
    last_eval = [None]

    def update_budgets(Cs):
        if last_eval[0] is None:
            last_eval[0] = [c.copy() for c in Cs]
            return
        subL, subR = [], []
        for b in range(nov):
            d2 = ((Cs[b] - last_eval[0][b]).ravel()) ** 2
            subL.append(np.array([math.sqrt(float(d2[lflat[si]].sum()))
                                  for si in range(nsec)]))
            subR.append(np.array([math.sqrt(float(d2[rflat[si]].sum()))
                                  for si in range(nsec)]))
        for w in range(nw):
            if w >= 1:
                budgets[w] += subL[w - 1]
            if w <= nw - 2:
                budgets[w] += subR[w]
        last_eval[0] = [c.copy() for c in Cs]

    def solve_sector(w, si, Cs):
        M = base_sec(w, si)
        if w >= 1 or w <= nw - 2:
            M = M.copy()
            if w >= 1:
                M += Cs[w - 1][np.ix_(lidx[si], lidx[si])] * lmask[si]
            if w <= nw - 2:
                M -= Cs[w][np.ix_(ridx[si], ridx[si])] * rmask[si]
        lam, vec = dense_eigh(M, subset_by_index=[0, 0])
        lam_cache[w][si] = float(lam[0])
        budgets[w][si] = 0.0
        return float(lam[0]), vec[:, 0]

    def ground(w, Cs):
        optimistic = [(lam_cache[w][si] - budgets[w][si], si)
                      for si in range(nsec)]
        optimistic.sort()
        best = (math.inf, None, None)
        for opt, si in optimistic:
            if opt >= best[0]:
                break
            lam, vec = solve_sector(w, si, Cs)
            if lam < best[0]:
                best = (lam, si, vec)
        v = np.zeros(dim)
        v[sec_ix[best[1]]] = best[2]
        return best[0], v

    def oracle(Cs):
        update_budgets(Cs)
        tot, const = 0.0, 0.0
        facs = [[] for _ in range(nov)]     # per overlap: (sign, D x E)
        for w in range(nw):
            lam0, v = ground(w, Cs)
            tot += lam0
            const += float(v @ (mats[w] @ v))
            if w >= 1:
                facs[w - 1].append((1.0, v.reshape(D, E).copy()))
            if w <= nw - 2:
                facs[w].append((-1.0, np.ascontiguousarray(
                    v.reshape(E, D).T)))
        return tot, const, facs

    def cut_dot(f1, f2):
        tot = 0.0
        for b in range(nov):
            for s1, A in f1[b]:
                for s2, B in f2[b]:
                    tot += s1 * s2 * float(np.sum((A.T @ B) ** 2))
        return tot

    def cut_dot_C(f, Cs):
        tot = 0.0
        for b in range(nov):
            for s1, A in f[b]:
                tot += s1 * float(np.sum(A * (Cs[b] @ A)))
        return tot

    # C restricted to sector-conserving form: WLOG by symmetrization
    # (conjugating by e^{i theta N} leaves the objective invariant), and
    # it keeps the correction krons sparse for sectored certification
    cmask = _sector_mask(round(math.log(D) / math.log(4)))
    Cs = [np.zeros((D, D)) for _ in range(nov)]
    tau = 1.0
    fref, a0, f0 = oracle(Cs)
    A, F = [a0], [f0]
    gram = np.array([[cut_dot(f0, f0)]])
    best = (fref, [c.copy() for c in Cs])
    for _ in range(iters - 1):
        m = len(A)
        b = np.array([A[i] + cut_dot_C(F[i], Cs) for i in range(m)])
        mu = np.full(m, 1.0 / m)
        eta = 1.0 / (1.0 + tau * float(np.max(np.abs(gram))))
        for _ in range(250):
            grad = b + tau * (gram @ mu)
            mu = mu * np.exp(-eta * (grad - grad @ mu))
            mu /= mu.sum()
        Cnew = [c.copy() for c in Cs]
        for i in range(m):
            if mu[i] < 1e-12:
                continue
            for bb in range(nov):
                for s1, Af in F[i][bb]:
                    Cnew[bb] += (tau * mu[i] * s1) * (Af @ Af.T)
        for bb in range(nov):
            Cnew[bb] *= cmask
        fnew, an, fn = oracle(Cnew)
        A.append(an)
        F.append(fn)
        row = np.array([cut_dot(fn, Fi) for Fi in F])
        gram = np.block([[gram, row[:-1, None]],
                         [row[None, :-1], np.array([[row[-1]]])]])
        if len(A) > 40:
            A, F = A[-40:], F[-40:]
            gram = gram[-40:, -40:]
        if fnew > fref:
            Cs, fref, tau = Cnew, fnew, min(tau * 1.4, 50.0)
        else:
            tau = max(tau * 0.6, 1e-3)
        if fref > best[0]:
            best = (fref, [c.copy() for c in Cs])
    return best[1]


def h_chain_bracket(n: int, d: float = 1.8, ell: int = 3,
                    correction_iters: int = 60,
                    cs_rounds: int = 2, fp32: bool = False) -> Certified:
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
                        # reverse for Y'Y). |g| only, no sign bookkeeping.
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
        from scipy import sparse
        acc = [np.zeros(dim) for _ in range(nw)]
        for k, (c, _, _) in enumerate(cs_terms):
            for w, wt, dv in sides[k][0]:
                acc[w] -= c * eps_v[k] * wt * dv
            for w, wt, dv in sides[k][1]:
                acc[w] -= c / eps_v[k] * wt * dv
        return [(base[w] + sparse.diags(acc[w])).tocsr()
                for w in range(nw)]

    # the naive balance update eps* = sqrt(<R>/<L>) is NOT an ascent step
    # (it optimizes against the current minimizers; lambda_min then
    # re-minimizes and can drop, measured). Damped updates plus
    # best-by-measured-total tracking make the loop monotone-safe: the
    # eps=1 start is included, so it can never end worse than it began.
    best_eps, best_tot = eps.copy(), -math.inf
    for _ in range(cs_rounds):
        mats = assemble(eps)
        tot, p2 = 0.0, []
        for M in mats:
            lam0, v = _sectored_ground(M)
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
        tot = sum(_sectored_ground(M)[0] for M in mats)
        if tot < best_tot:
            eps, mats = best_eps, assemble(best_eps)
    else:
        mats = assemble(eps)
    if correction_iters and nw > 1:
        from scipy import sparse as sp
        D = 4 ** (ell - 1)
        Cs = _window_multipliers(mats, D, correction_iters)
        IE = sp.identity(4 ** ell // D, format="csr")
        out = []
        for w in range(nw):
            M = mats[w]
            if w > 0:
                M = M + sp.kron(sp.csr_matrix(Cs[w - 1]), IE, format="csr")
            if w < nw - 1:
                M = M - sp.kron(IE, sp.csr_matrix(Cs[w]), format="csr")
            out.append(M.tocsr())
        mats = out
    for M in mats:
        lo, _up = _eigen_bracket_sectored(M, fp32=fp32)
        lower += lo

    # upper: product of exactly solved blocks, cross energy by exact
    # factorization of block-diagonal spin-orbital 1-RDMs
    from scipy.sparse.linalg import eigsh

    def product_upper(sizes):
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
            Hb = _fock_hamiltonian(np.eye(len(b)), h_own, eri_b, enuc_b,
                                   dense=False)
            _, Vec = eigsh(Hb, k=1, which="SA",
                           v0=_arpack_v0(Hb.shape[0]))
            v = Vec[:, 0] / np.linalg.norm(Vec[:, 0])
            e = float(v @ (Hb @ v))
            upper += e + 8 * (Hb.shape[0] + 2) * np.finfo(float).eps \
                * (float(np.abs(Hb).sum(axis=0).max()) + abs(e))
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
                            # the spin-free two-electron operator contracts
                            # p with r at one spin and q with s at another,
                            # independently, so the two spin indices run
                            # CROSSED through the pair of blocks. Matched
                            # spins alone would drop every term an
                            # alpha-beta coherence contributes, and an
                            # odd-atom block has that coherence: its ground
                            # state is a spin doublet and any member of it
                            # will do. Measured at n=6, dropping them put
                            # this bound 1.1 mHa above the product state's
                            # own energy at ell=3 -- loose, not wrong, and
                            # invisible to every containment test.
                            for sp in range(2):
                                for sq in range(2):
                                    upper -= 0.5 * g \
                                        * gA[2 * ip + sp, 2 * is2 + sq] \
                                        * gB[2 * iq + sq, 2 * ir + sp]
        # the trial state's electron count, so a caller pairing this
        # upper half with a fixed-N lower bound can check rather than
        # assume that the two are talking about the same sector. Each
        # block ground state is taken over its whole Fock space, so half
        # filling is a measured property of these integrals and not a
        # theorem.
        nel = sum(float(np.trace(gammas[bi])) for bi in range(len(blocks)))
        return upper, nel

    upper, nelec = min((product_upper(s)
                        for s in _block_tilings(n, ell, merge_lone=False)),
                       key=lambda t: t[0])
    return Certified(0.5 * (upper + lower), 0.5 * (upper - lower),
                     Tier.RIGOROUS,
                     (f"h-chain marginal-lower ell={ell} n={n} d={d:g} "
                      f"iters={correction_iters} block-product-upper "
                      f"upper-nelec={nelec:.6f}",))


# --------------------------------------------- the 2-RDM lower bound. The
# window ladder above prices a lower bound at 4^ell and stops where the
# matrix stops being formable. This one is priced in orbitals instead of
# in states, so its cost grows polynomially where the other grows
# exponentially.
#
# The energy is a linear functional of the two-particle density matrix,
# so minimizing it over a set that CONTAINS every N-representable 2-RDM
# gives a lower bound on the ground energy. Deciding N-representability
# is itself hard, so the containing set is the usual 2-positivity
# relaxation: the particle-particle block D, the hole-hole block Q and
# the particle-hole block G are each required to be PSD, since each is a
# Gram matrix of operators acting on the state. Q and G are affine in D,
# and the one-particle matrix is a contraction of D, so the whole thing
# is one semidefinite program in D alone.
#
# The certificate does not come from the solver. For any Y_Q >= 0 and
# Y_G >= 0 the identity
#
#     E(D) - enuc = <W, D> + <Y_Q, Q(D)> + <Y_G, G(D)> + c0
#
# holds with W = W0 - A_Q*(Y_Q) - A_G*(Y_G), and both inner products are
# non-negative on the feasible set, so
#
#     E(D) - enuc >= <W, D> + c0 >= lambda_min(W) * Tr(D) + c0.
#
# Tr(D) is fixed at N(N-1)/2, so a lower bound on lambda_min(W) is a
# lower bound on the energy. SCS proposes Y_Q and Y_G and is not
# trusted: a bad proposal moves W and loosens the bound, and cannot
# make it wrong. Every positivity fact is then issued by eigen_bracket,
# which already carries its own floating-point margins, so this rewrite
# certifies by composition rather than by a new proof.
#
# Everything here is blocked by spin projection. A geminal a_q a_p
# carries S_z = sigma_p + sigma_q, and a particle-hole operator
# a_dag_q a_p carries S_z = sigma_p - sigma_q, so on a state of definite
# S_z each matrix is zero between labels that differ. The Hamiltonian
# commutes with S_z, so the ground energy of the N-electron sector is
# attained at a state of definite S_z, which means the true 2-RDM stays
# inside the blocked feasible set and the bound stays a bound. Blocking
# can only tighten it, since it minimizes over less.


def _rdm2_spin_block(h, eri):
    """Spatial orbitals to spin orbitals, p_so = 2*p + spin. Returns the
    spin-orbital core matrix and the physicists' <PQ|RS>, which is the
    chemists' (PR|QS) with the two spin deltas the Coulomb operator
    carries."""
    n = len(h)
    nso = 2 * n
    hs = np.zeros((nso, nso))
    for p in range(n):
        for q in range(n):
            for s in range(2):
                hs[2 * p + s, 2 * q + s] = h[p, q]
    g = np.zeros((nso,) * 4)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    for sa in range(2):
                        for sb in range(2):
                            g[2 * p + sa, 2 * q + sb,
                              2 * r + sa, 2 * s + sb] = eri[p, r, q, s]
    return hs, g


def _rdm2_layout(labels):
    """Group indices by symmetry label and lay the blocks out end to end
    in one vector. Returns the block sizes, the offset of each block's
    vec, the label of each index, its position inside its own block, and
    the total length."""
    order = sorted(set(labels))
    pos, lab_of, sizes, off, total = {}, {}, [], {}, 0
    for lab in order:
        members = [i for i, x in enumerate(labels) if x == lab]
        off[lab] = total
        sizes.append((lab, len(members)))
        for k, i in enumerate(members):
            pos[i] = k
            lab_of[i] = lab
        total += len(members) ** 2
    return dict(sizes=sizes, off=off, pos=pos, lab=lab_of, total=total,
                width={lab: m for lab, m in sizes})


@functools.lru_cache(maxsize=8)
def _rdm2_maps(nso, N):
    """The exact affine maps from the 2-RDM to everything else, built
    once per (nso, N) and cached because they depend on no integral.

    D lives in the geminal basis: rows and columns are ordered pairs
    p < q, and D_IJ = <(a_q a_p)^dagger (a_s a_r)> is a Gram matrix,
    hence PSD. Spin projection splits it into three blocks, and Q with
    it; G splits into three of its own. Each block is stored as its own
    square matrix and the blocks are laid end to end in one vector, so
    the maps below are sparse operators on that vector.

    All three identities were checked against exactly computed RDMs
    before this code was written, at (nso, N) = (4, 2), (6, 3), (6, 2)
    and (8, 4), and they agree to 1e-15:

        1D_pr        = sum_q 2D_pq,rq / (N - 1)
        2Q_pq,rs     = 2D_pq,rs + d_pr d_qs - d_ps d_qr
                       - d_qs 1D_rp + d_ps 1D_rq
                       + d_qr 1D_sp - d_pr 1D_sq
        2G_pq,rs     = d_qs 1D_pr - 2D_ps,rq
    """
    import scipy.sparse as sp
    if N < 2:
        raise ValueError("the 2-RDM needs at least two particles")

    def spin(p):
        return p % 2

    pairs = [(p, q) for p in range(nso) for q in range(p + 1, nso)]
    pidx = {}
    for k, (p, q) in enumerate(pairs):
        pidx[(p, q)] = (k, 1.0)
        pidx[(q, p)] = (k, -1.0)
    dl = _rdm2_layout([spin(p) + spin(q) for p, q in pairs])

    ph = [(p, q) for p in range(nso) for q in range(nso)]
    gl = _rdm2_layout([spin(p) - spin(q) for p, q in ph])

    def dcol(i, j):
        """Column of D[i, j] in the blocked vector, or None when the two
        geminals carry different S_z and the entry is structurally
        zero."""
        a, b = dl["lab"][i], dl["lab"][j]
        if a != b:
            return None
        m = dl["width"][a]
        return dl["off"][a] + dl["pos"][i] + m * dl["pos"][j]

    def d2row(p, q, r, s):
        a, b = pidx.get((p, q)), pidx.get((r, s))
        if a is None or b is None:
            return {}
        c = dcol(a[0], b[0])
        return {} if c is None else {c: a[1] * b[1]}

    def add(dst, src, w):
        for k, v in src.items():
            dst[k] = dst.get(k, 0.0) + w * v

    one = {}                          # 1D_pr as a row on the blocked vec
    f = 1.0 / (N - 1)
    for p in range(nso):
        for r in range(nso):
            row = {}
            for q in range(nso):
                a, b = pidx.get((p, q)), pidx.get((r, q))
                if a is None or b is None:
                    continue
                c = dcol(a[0], b[0])
                if c is not None:
                    row[c] = row.get(c, 0.0) + f * a[1] * b[1]
            one[p * nso + r] = row

    qA, qc = {}, np.zeros(dl["total"])
    for I, (p, q) in enumerate(pairs):
        for J, (r, s) in enumerate(pairs):
            rowcol = dcol(I, J)
            if rowcol is None:
                continue
            row = dict(d2row(p, q, r, s))
            if q == s:
                add(row, one[r * nso + p], -1.0)
            if p == s:
                add(row, one[r * nso + q], +1.0)
            if q == r:
                add(row, one[s * nso + p], +1.0)
            if p == r:
                add(row, one[s * nso + q], -1.0)
            qA[rowcol] = row
            qc[rowcol] = (1.0 if (p == r and q == s) else 0.0) \
                - (1.0 if (p == s and q == r) else 0.0)

    def gcol(i, j):
        a, b = gl["lab"][i], gl["lab"][j]
        if a != b:
            return None
        m = gl["width"][a]
        return gl["off"][a] + gl["pos"][i] + m * gl["pos"][j]

    gA = {}
    for I, (p, q) in enumerate(ph):
        for J, (r, s) in enumerate(ph):
            rowcol = gcol(I, J)
            if rowcol is None:
                continue
            row = {}
            if q == s:
                add(row, one[p * nso + r], +1.0)
            add(row, d2row(p, s, r, q), -1.0)
            gA[rowcol] = row

    def spmat(rows, nrows):
        r, c, v = [], [], []
        for i, row in rows.items():
            for j, val in row.items():
                if val:
                    r.append(i)
                    c.append(j)
                    v.append(val)
        return sp.csr_matrix((v, (r, c)), shape=(nrows, dl["total"]))

    return dict(pairs=pairs, ph=ph, nso=nso, dl=dl, gl=gl, one=one,
                AQ=spmat(qA, dl["total"]), AG=spmat(gA, gl["total"]),
                qc=qc, M=len(pairs), n2=len(ph))


def _rdm2_blocks(mp, vec, which="dl"):
    """Cut a blocked vector into its square matrices."""
    lay = mp[which]
    return [vec[lay["off"][lab]:lay["off"][lab] + m * m].reshape(
        (m, m), order="F") for lab, m in lay["sizes"] if m]


def _rdm2_energy_operator(mp, hs, g):
    """W0 with E - enuc = <W0, D>, as a blocked vector. The one-body part
    rides in through the contraction that makes 1D a functional of D, and
    the two-body part is the antisymmetrized integral in the geminal
    basis. Entries between different S_z labels are dropped because D is
    zero there."""
    nso, dl = mp["nso"], mp["dl"]
    W = np.zeros(dl["total"])
    for p in range(nso):
        for r in range(nso):
            hv = hs[p, r]
            if hv:
                for k, val in mp["one"][p * nso + r].items():
                    W[k] += hv * val
    for I, (p, q) in enumerate(mp["pairs"]):
        for J, (r, s) in enumerate(mp["pairs"]):
            if dl["lab"][I] != dl["lab"][J]:
                continue
            m = dl["width"][dl["lab"][I]]
            k = dl["off"][dl["lab"][I]] + dl["pos"][I] + m * dl["pos"][J]
            W[k] += g[p, q, r, s] - g[p, q, s, r]
    # symmetrize each block
    out = []
    for B in _rdm2_blocks(mp, W):
        out.append(0.5 * (B + B.T))
    return out


def _rdm2_propose(mp, W0, T, eps, max_iters, conditions="DQG"):
    """The untrusted half. SCS minimizes the energy over the blocked
    relaxation, and its dual variables for the Q and G blocks are the
    multipliers the certificate wants. Nothing here is believed: the
    returned matrices are only a starting point, and the caller shifts
    them until eigen_bracket says they are PSD.

    conditions selects which blocks are required PSD, and it is the
    rewrite's ladder. D alone is the cheapest and loosest, since with
    only the trace fixed the minimum collapses to the smallest
    eigenvalue of the two-electron operator. Adding Q and then G costs
    more cone and buys a tighter bound. Dropping a condition only
    enlarges the feasible set, so every rung is still a lower bound."""
    import cvxpy as cp
    dl, gl = mp["dl"], mp["gl"]
    dsz = [m for _, m in dl["sizes"] if m]
    gsz = [m for _, m in gl["sizes"] if m]
    blocks = [cp.Variable((m, m), symmetric=True) for m in dsz]

    def vecF(X):
        try:
            return cp.vec(X, order="F")
        except TypeError:                        # cvxpy without order=
            return cp.vec(X)

    def reshapeF(x, shape):
        try:
            return cp.reshape(x, shape, order="F")
        except TypeError:
            return cp.reshape(x, shape)

    dv = cp.hstack([vecF(B) for B in blocks])
    q = mp["AQ"] @ dv + mp["qc"]
    gvec = mp["AG"] @ dv
    cons = [B >> 0 for B in blocks]
    for lab, m in dl["sizes"]:
        if m and "Q" in conditions:
            cons.append(reshapeF(q[dl["off"][lab]:dl["off"][lab] + m * m],
                                 (m, m)) >> 0)
    for lab, m in gl["sizes"]:
        if m and "G" in conditions:
            cons.append(reshapeF(gvec[gl["off"][lab]:gl["off"][lab] + m * m],
                                 (m, m)) >> 0)
    cons.append(sum(cp.trace(B) for B in blocks) == T)
    obj = sum(cp.sum(cp.multiply(W, B)) for W, B in zip(W0, blocks))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=max_iters)

    nd = len(dsz)
    nq = len(dsz) if "Q" in conditions else 0
    ng = len(gsz) if "G" in conditions else 0
    YQ = [c.dual_value for c in cons[nd:nd + nq]]
    YG = [c.dual_value for c in cons[nd + nq:nd + nq + ng]]
    if any(y is None for y in YQ + YG):
        raise ValueError(f"SCS returned no dual ({prob.status})")
    return ([np.asarray(y) for y in YQ], [np.asarray(y) for y in YG],
            prob.status, float(prob.value))


def _rdm2_psd_shift(Y):
    """Return Y + delta*I with lambda_min certified non-negative, and the
    shift it took. eigen_bracket is the arbiter, and it runs on the
    matrix that is actually used rather than on the one proposed."""
    Y = np.asarray(Y, float)
    Y = 0.5 * (Y + Y.T)
    delta = 0.0
    for _ in range(60):
        Ys = Y + delta * np.eye(len(Y))
        c = eigen_bracket(Ys)
        lo = c.value - c.err
        if lo >= 0.0:
            return Ys, delta
        delta += max(-lo * 1.5, 1e-15 * (1.0 + abs(c.value)))
    raise ValueError("multiplier could not be certified PSD")


def _rdm2_pack(mp, mats, which="dl"):
    lay = mp[which]
    out = np.zeros(lay["total"])
    if not mats:                 # a condition the ladder did not impose
        return out
    i = 0
    for lab, m in lay["sizes"]:
        if not m:
            continue
        out[lay["off"][lab]:lay["off"][lab] + m * m] = \
            mats[i].reshape(-1, order="F")
        i += 1
    return out


def _rdm2_assemble(mp, W0, YQ, YG):
    """W and c0 in floating point, with a Frobenius bound on their own
    rounding. The pad is carried rather than assumed away, because
    |<A, D>| <= ||A||_2 Tr(D) whenever D is PSD, so an error of ||E||_F
    in W costs at most ||E||_F * Tr(D) in the bound."""
    AQ, AG = mp["AQ"], mp["AG"]
    yq = _rdm2_pack(mp, YQ, "dl")
    yg = _rdm2_pack(mp, YG, "gl")
    w0 = _rdm2_pack(mp, W0, "dl")
    W = w0 - (AQ.T @ yq) - (AG.T @ yg)
    S = np.abs(w0) + (abs(AQ).T @ np.abs(yq)) + (abs(AG).T @ np.abs(yg))
    K = 1.0 + np.asarray(abs(AQ).sign().T @ np.ones(AQ.shape[0])).ravel() \
        + np.asarray(abs(AG).sign().T @ np.ones(AG.shape[0])).ravel()
    u = float(np.finfo(np.float64).eps) / 2
    k = float(K.max()) + 2.0
    gam = k * u / (1.0 - k * u) if k * u < 0.5 else np.inf
    padW = float(np.sqrt(((gam * S) ** 2).sum()))

    c0 = -float(yq @ mp["qc"])
    padc = gam * float(np.abs(yq) @ np.abs(mp["qc"]))
    blocks = [0.5 * (B + B.T) for B in _rdm2_blocks(mp, W)]
    return blocks, padW, c0, padc


def _rdm2_determinant_upper(h, eri, N, scf_iters: int = 200):
    """A rigorous variational upper bound from one closed-shell
    determinant. Any orthonormal set of orbitals gives a valid bound, so
    convergence is not part of the claim and a stopped iteration is as
    honest as a converged one. Self-consistency only chooses better
    orbitals, and it is worth doing here because the upper half is the
    binding side of every 2-RDM bracket measured so far. scf_iters=0
    falls back to the core-Hamiltonian guess.

    A caller holding a better upper bound should pass it in instead, and
    the bracket tightens on the side that needs it."""
    if N % 2:
        raise ValueError("the built-in upper bound is closed-shell only")
    h = np.asarray(h, float)
    eri = np.asarray(eri, float)
    nocc = N // 2
    if scf_iters:
        C, its, conv = _rhf(h, eri, N, max_iters=scf_iters)
        note = f"scf={its}it" + ("" if conv else "-unconverged")
    else:
        _, C = np.linalg.eigh(h)
        note = "core-guess"
    occ = C[:, :nocc]
    hm = occ.T @ h @ occ
    e = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, occ, occ, occ, occ)
    d = np.arange(N // 2)
    energy = float(
        2 * np.trace(hm)
        + 2 * e[d[:, None], d[:, None], d[None, :], d[None, :]].sum()
        - e[d[:, None], d[None, :], d[None, :], d[:, None]].sum())
    return energy, note


def _rdm2_lower(h, eri, N, enuc, eps, max_iters, conditions="DQG"):
    """The lower half on its own, returned as (bound, provenance).

    It is separate because it does not depend on whatever upper bound it
    will be paired with, so a caller that tries several upper bounds
    pays for the semidefinite program once. conditions is the ladder,
    from D through DQ to DQG, cheapest and loosest first."""
    h = np.asarray(h, float)
    eri = np.asarray(eri, float)
    hs, g = _rdm2_spin_block(h, eri)
    nso = len(hs)
    mp = _rdm2_maps(nso, N)
    T = N * (N - 1) / 2.0
    W0 = _rdm2_energy_operator(mp, hs, g)

    YQ, YG, status, _ = _rdm2_propose(mp, W0, T, eps, max_iters,
                                      conditions)
    shifts = []
    certified = []
    for group in (YQ, YG):
        out = []
        for Y in group:
            Ys, d = _rdm2_psd_shift(Y)
            out.append(Ys)
            shifts.append(d)
        certified.append(out)
    Wb, padW, c0, padc = _rdm2_assemble(mp, W0, certified[0], certified[1])

    # lambda_min of a block-diagonal matrix is the smallest over blocks,
    # and each block gets its own certified bracket
    mu = None
    for B in Wb:
        c = eigen_bracket(B)
        lo = c.value - c.err
        mu = lo if mu is None else min(mu, lo)
    lower = (mu - padW) * T + c0 - padc + enuc

    blocks = "+".join(str(m) for _, m in mp["dl"]["sizes"] if m)
    return lower, (f"rdm2-{conditions} N={N} nso={nso} D-blocks={blocks} "
                   f"scs={status} shift={max(shifts, default=0.0):.1e} "
                   f"pad=({padW:.1e},{padc:.1e}) +fp")


def rdm2_energy_bracket(h, eri, N: int, enuc: float = 0.0,
                        upper: float = None, eps: float = 1e-8,
                        max_iters: int = 100_000,
                        lower: Tuple[float, str] = None,
                        conditions: str = "DQG") -> Certified:
    """Certified bracket on the ground energy of the N-electron sector,
    with the lower half from the 2-positivity relaxation of the 2-RDM.

    h and eri are spatial-orbital integrals in an ORTHONORMAL basis, eri
    in chemists' (pq|rs). The claim is about those matrices exactly as
    stored, which is the same declaration the eigenvalue brackets make.

    Two things separate this from the window ladder, and both are worth
    stating. Its cost is polynomial in orbitals rather than exponential
    in states. And it certifies the N-electron sector rather than the
    whole Fock space, which is a narrower question, so the two rewrites
    do not bracket quite the same number. Where the global ground state
    is known to carry N electrons the two agree, and knowing that is a
    separate claim this function does not make.

    The upper half is a single determinant unless the caller supplies a
    better one, and it is usually the loose side by a wide margin. Pass
    lower to reuse a bound _rdm2_lower already computed, which is what a
    planner racing several upper bounds against one lower does.
    """
    lo, prov = _rdm2_lower(h, eri, N, enuc, eps, max_iters, conditions) \
        if lower is None else lower
    if upper is None:
        det, note = _rdm2_determinant_upper(h, eri, N)
        up = det + enuc
    else:
        up, note = float(upper), "given"
    if up < lo:
        raise ValueError(
            f"2-RDM lower bound {lo:.9f} exceeds the upper bound "
            f"{up:.9f}; the upper bound is not variational for N={N}")
    return Certified(0.5 * (up + lo), 0.5 * (up - lo), Tier.RIGOROUS,
                     (f"{prov} upper={note}",))


def h_chain_rdm2_bracket(n: int, d: float = 1.8, ell: int = 3,
                         eps: float = 1e-8,
                         max_iters: int = 100_000,
                         window: Certified = None,
                         lower: Tuple[float, str] = None) -> Certified:
    """The two halves of one bracket, from two unrelated proofs.

    The upper half is the window ladder's product of block ground
    states, which is a variational trial state. The lower half is the
    2-positivity relaxation of the 2-RDM. That is the sandwich the
    chemistry column of TARGETS describes, and neither half knows the
    other exists.

    The halves have to be talking about the same sector or the sandwich
    is not one. The 2-RDM bound is about the N-electron sector, and the
    window's trial state is a product of block ground states taken over
    each block's whole Fock space, so its electron count is whatever the
    blocks prefer. Measured on hydrogen chains at d = 1.8 every block
    comes out exactly half filled, but that is a property of these
    integrals rather than a theorem, so it is read off the window
    certificate and checked here. A mismatch refuses.

    The window's own lower half bounds the ground energy over all
    sectors, which also bounds the N-electron one, so the two lower
    bounds intersect and the tighter is kept.
    """
    win = h_chain_bracket(n, d, ell) if window is None else window
    tag = "upper-nelec="
    prov = win.provenance[0]
    if tag not in prov:
        raise ValueError("the window certificate does not report its "
                         "trial state's electron count")
    nelec = float(prov[prov.index(tag) + len(tag):].split()[0])
    if abs(nelec - n) > 1e-6:
        raise ValueError(
            f"the window trial state holds {nelec:.6f} electrons, not "
            f"{n}, so its energy does not bound the {n}-electron sector")

    T, V, eri, enuc = _h_chain_basis(n, d)
    h = T + V.sum(0)
    # two upper bounds are in hand and both are variational for N
    # electrons, so the better one is kept. Measured, the self-consistent
    # determinant beats the block product at these sizes, 62 mHa above
    # exact against 147 at H4, and it costs no window at all.
    det = _rdm2_determinant_upper(h, eri, n)[0] + enuc
    up = min(win.value + win.err, det)
    upper_from = "determinant" if det < win.value + win.err else "window"
    c = rdm2_energy_bracket(h, eri, n, enuc, upper=up,
                            eps=eps, max_iters=max_iters, lower=lower)
    lower = max(c.value - c.err, win.value - win.err)
    binding = "2rdm" if c.value - c.err >= win.value - win.err else "window"
    return Certified(0.5 * (up + lower), 0.5 * (up - lower),
                     min(win.tier, c.tier),
                     win.provenance + c.provenance
                     + (f"intersect-lower binding={binding} "
                        f"upper={upper_from}",),
                     min(1.0, win.fail_p + c.fail_p))


# ------------------------------------------------------- a real basis set.
# STO-3G for the first-row elements this library needs, in the published
# contracted form. The contraction coefficients are shared across
# elements because the fit is to a Slater orbital of exponent 1 and only
# the exponents are scaled, by zeta squared.
#
# Recalled constants are worth exactly what they are checked against, so
# these are gated on molecular energies nobody here chose: RHF/STO-3G for
# water, methane and ammonia, which between them exercise all four
# elements. Measured -74.9631, -39.7268 and -55.4541 against literature
# values of about -74.9659, -39.727 and -55.4554. A test pins all three.
#
# One thing did not resolve. N2 comes out at -106.766 where the value
# recalled for it was -107.496. Ammonia constrains the nitrogen
# exponents to about half a per cent, measured by perturbing them, and a
# basis error large enough to move N2 by 0.73 hartree is far outside
# that, so the recollection is the more likely wrong one. It is recorded
# rather than resolved.

_STO3G_1S = (0.15432897, 0.53532814, 0.44463454)
_STO3G_2S = (-0.09996723, 0.39951283, 0.70011547)
_STO3G_2P = (0.15591627, 0.60768372, 0.39195739)
_STO3G_EXPONENTS = {
    "H": {"1s": (3.42525091, 0.62391373, 0.16885540)},
    "C": {"1s": (71.6168370, 13.0450960, 3.5305122),
          "2sp": (2.9412494, 0.6834831, 0.2222899)},
    "N": {"1s": (99.1061690, 18.0523120, 4.8856602),
          "2sp": (3.7804559, 0.8784966, 0.2857144)},
    "O": {"1s": (130.7093200, 23.8088610, 6.4436083),
          "2sp": (5.0331513, 1.1695961, 0.3803890)},
}
_ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8}
BOHR_PER_ANGSTROM = 1.8897261254578281


def sto3g(molecule, angstrom: bool = False):
    """(atoms, shells, nelec) for a molecule given as [(element, xyz)].

    Coordinates are bohr unless angstrom is set. Every element carries
    its 1s, and the first-row ones also carry a 2s and the three 2p
    components, which is what STO-3G means for them. Shells come back one
    Cartesian AO each, which is what _md_integrals takes.

    nelec is the neutral electron count, so a caller wanting an ion
    should pass its own."""
    atoms, shells, nelec = [], [], 0
    for el, xyz in molecule:
        if el not in _STO3G_EXPONENTS:
            raise ValueError(f"no STO-3G data here for {el!r}; this "
                             f"library carries "
                             f"{sorted(_STO3G_EXPONENTS)}")
        p = tuple(float(c) * (BOHR_PER_ANGSTROM if angstrom else 1.0)
                  for c in xyz)
        atoms.append((_ELEMENT_Z[el], p))
        nelec += _ELEMENT_Z[el]
        sets = _STO3G_EXPONENTS[el]
        shells.append((p, (0, 0, 0), tuple(zip(sets["1s"], _STO3G_1S))))
        if "2sp" in sets:
            shells.append((p, (0, 0, 0),
                           tuple(zip(sets["2sp"], _STO3G_2S))))
            for l in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                shells.append((p, l, tuple(zip(sets["2sp"], _STO3G_2P))))
    return atoms, shells, nelec


def _rhf(h, eri, N, max_iters: int = 200, tol: float = 1e-10):
    """Closed-shell self-consistent field in an orthonormal basis.

    Returns (orbitals, iterations, converged). Convergence is not part of
    any claim downstream, because the determinant built from ANY
    orthonormal orbitals is variational, so a stopped iteration gives a
    looser upper bound and never an invalid one. It is reported so a
    reader can tell a converged bound from a stopped one."""
    h = np.asarray(h, float)
    eri = np.asarray(eri, float)
    nocc = N // 2
    _, C = np.linalg.eigh(h)
    E = None
    for it in range(max_iters):
        Co = C[:, :nocc]
        P = 2.0 * Co @ Co.T
        F = h + np.einsum("rs,pqrs->pq", P, eri) \
            - 0.5 * np.einsum("rs,prqs->pq", P, eri)
        Enew = 0.5 * float(np.sum(P * (h + F)))
        _, C = np.linalg.eigh(F)
        if E is not None and abs(Enew - E) < tol:
            return C, it + 1, True
        E = Enew
    return C, max_iters, False


def molecular_integrals(atoms, shells):
    """(h, eri, enuc) in an orthonormal spatial-orbital basis, from
    Cartesian Gaussians of arbitrary angular momentum.

    atoms and shells are what _md_integrals takes: [(Z, xyz)], and one
    AO per shell as (xyz, (i, j, k), [(exponent, contraction), ...]).
    Lowdin absorbs the overlap, so the AOs need not be normalized, and
    eri comes back in chemists' (pq|rs), which is what every rewrite
    downstream expects."""
    S, h, eri, enuc = _md_integrals(atoms, shells)
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T
    return (X @ h @ X,
            np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, X, X, X, X), enuc)


def _sector_dim(nao, nelec):
    return math.comb(2 * nao, nelec)


def molecule_dense_bracket(atoms, shells, nelec: int) -> Certified:
    """The exact answer, by diagonalizing the N-electron sector of the
    Fock space. Particle number is conserved, so the sector is a block
    and there is no need to carry the rest: at eight orbitals the whole
    space is 65,536 states and the six-electron sector is 8,008. The
    bracket is eigen_bracket's, with its floating-point margins."""
    from scipy import sparse                            # noqa: F401
    nao = len(shells)
    H = _fock_hamiltonian(*_md_integrals(atoms, shells), dense=False)
    occ = np.array([bin(i).count("1") for i in range(H.shape[0])])
    idx = np.flatnonzero(occ == nelec)
    sub = np.asarray(H[idx][:, idx].todense())
    c = eigen_bracket(sub)
    return replace(c, provenance=(
        f"molecule-dense-sector nao={nao} N={nelec} dim={len(idx)} "
        + c.provenance[0],))


def _rdm2_rewrite(h, eri, nelec, enuc, nao, eps, max_iters):
    """The 2-RDM as a Rewrite, with the conditions as its ladder.

    D alone fixes only the trace, so its minimum is the smallest
    eigenvalue of the two-electron operator and it is very loose. Adding
    Q and then G costs more cone and buys a tighter bound, and dropping
    a condition only enlarges the feasible set, so every rung is still a
    lower bound. Measured on hydrogen chains at n = 4 to 8, the half
    width per atom runs 98 to 291 mHa at D, 13 to 54 at DQ, and 8.0 to
    8.6 at DQG, while the cost runs about 0.2 s, 0.6 to 6 s, and 2.5 to
    65 s.

    The prices below are k n^4 with k measured over that range. They
    only order the attempts, and the receipt logs predicted against
    measured so the model is auditable."""
    price = {"D": 0.1, "DQ": 5.0, "DQG": 30.0}

    def run(cond):
        return rdm2_energy_bracket(h, eri, nelec, enuc, eps=eps,
                                   max_iters=max_iters, conditions=cond)

    return Rewrite("rdm2", ("D", "DQ", "DQG"),
                   lambda c: price[c] * nao ** 4, run)


def molecule_energy_dispatch(atoms, shells, nelec: int, tol: float,
                             dense_nao_max: int = 8,
                             dense_dim_max: int = 5000,
                             jump: bool = True, eps: float = 1e-7,
                             max_iters: int = 100_000) -> Certified:
    """Certified bracket on the N-electron ground energy of a molecule,
    from Cartesian Gaussians of any angular momentum, with two rewrites
    racing for it.

    The first diagonalizes the N-electron sector and is exact, at a cost
    that is the sector dimension cubed. The second relaxes the 2-RDM and
    is priced in orbitals. Polarization functions are what make the race
    interesting, because a p shell on every atom doubles the orbital
    count and cubes nothing the first rewrite can afford: six hydrogens
    with s and p are twelve orbitals and a sector of 134,596, which is
    past forming, while the relaxation there is the same size as a
    twelve-orbital problem anywhere else.

    Both rewrites answer the same question. The dense one brackets the
    sector directly; the relaxation bounds it from below and a
    self-consistent determinant from above, and that determinant is an
    N-electron state, so both brackets are about the N-electron ground
    energy and not about the whole Fock space.

    The dense rewrite is capped twice because it pays twice. It
    assembles the whole 4^nao Fock operator before it can cut the sector
    out, and then it diagonalizes a dense matrix of the sector's size,
    so dense_nao_max bounds the assembly and dense_dim_max bounds the
    diagonalization. Both have to hold or the rewrite is not offered,
    and the refusal says which one stopped it. Pricing only the
    diagonalization was wrong and measurably so: at eight orbitals with
    two electrons the sector is 120 wide and takes no time at all, while
    the assembly it needs first takes five seconds.
    """
    if nelec % 2:
        raise ValueError("the determinant upper bound is closed-shell "
                         "only, so nelec must be even")
    nao = len(shells)
    dim = _sector_dim(nao, nelec)
    h, eri, enuc = molecular_integrals(atoms, shells)

    rewrites = []
    if nao <= dense_nao_max and dim <= dense_dim_max:
        rewrites.append(Rewrite(
            "dense", (dim,),
            lambda k: 4.0 ** nao / 1e3 + k ** 3 / 1e6,
            lambda k: molecule_dense_bracket(atoms, shells, nelec)))
    rewrites.append(_rdm2_rewrite(h, eri, nelec, enuc, nao, eps, max_iters))

    def beyond():
        if nao <= dense_nao_max and dim <= dense_dim_max:
            return ("the ladder is exhausted; the dense sector is already "
                    "exact, so a tighter answer needs a better basis, not "
                    "a better rewrite")
        if nao > dense_nao_max:
            return (f"the exact rewrite was not offered: {nao} orbitals "
                    f"means a 4^{nao} Fock operator to assemble, past "
                    f"dense_nao_max={dense_nao_max}")
        return (f"the exact rewrite was not offered: the {nelec}-electron "
                f"sector has dimension {dim:,}, past "
                f"dense_dim_max={dense_dim_max:,}")

    rewrites[-1] = replace(rewrites[-1], price_beyond=beyond)
    return plan("molecule-energy", tol, rewrites, jump=jump,
                context=f"nao={nao} N={nelec} sector={dim}")


def hydrogen_shells(positions, polarized: bool = False,
                    pz_exp: float = 1.1, full_p: bool = False):
    """STO-3G s shells on each hydrogen, optionally polarized.

    polarized adds the sigma component pz, which is the one that matters
    for a chain along z and is what h2_polarized_bracket uses. full_p
    adds px and py as well, which triples the added orbitals and is
    usually more basis than the demonstration needs.

    Polarization is what pushes a small molecule past the point where its
    Fock sector can be formed, which is where the relaxation earns its
    place: six hydrogens with s alone are six orbitals and a formable
    924-dimensional sector, and with pz they are twelve orbitals and a
    sector of 134,596 behind a Fock space of 16,777,216."""
    shells = [(tuple(p), (0, 0, 0), _STO3G_H_RAW) for p in positions]
    comps = ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if full_p else ((0, 0, 1),)
    if polarized:
        for p in positions:
            for lz in comps:
                shells.append((tuple(p), lz, ((pz_exp, 1.0),)))
    return shells


# --------------------------------------------- certified reduced bases /
# eigenvector continuation. For AFFINE operator families
# H(theta) = H0 + theta H1, two structural facts make certified
# parameter sweeps nearly free:
#   1. lambda_min(theta) = min_v [<v|H0|v> + theta <v|H1|v>] is an
#      infimum of affine functions of theta, hence CONCAVE, so certified
#      snapshot lower bounds give a rigorous lower bound at every theta
#      between them by the chord inequality. O(1) online, no solve.
#   2. The reduced-basis Rayleigh quotient is EXACT at k x k cost online:
#      with B an orthonormal snapshot basis, min-eig of B'(H0+theta H1)B
#      is the full-space Rayleigh quotient of the best basis vector,
#      a variational upper bound (this is eigenvector continuation's
#      engine, and near-critical points it is startlingly accurate).
# Outside the snapshot hull concavity certifies nothing: refuse.
# One rewrite, two TARGETS columns: CEM parameter sweeps and the
# nuclear-EFT eigenvector-continuation program.


def tfi_chain(N):
    """(H0, H1) for the transverse-field Ising chain
    H(g) = -sum Z_i Z_{i+1} - g sum X_i. This is the demo affine family,
    swept across its quantum phase transition at g = 1."""
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.diag([1.0, -1.0])

    def embed(op, i):
        M = np.eye(1)
        for j in range(N):
            M = np.kron(M, op if j == i else np.eye(2))
        return M

    H0 = np.zeros((2 ** N, 2 ** N))
    for i in range(N - 1):
        H0 -= embed(Z, i) @ embed(Z, i + 1)
    H1 = np.zeros((2 ** N, 2 ** N))
    for i in range(N):
        H1 -= embed(X, i)
    return H0, H1


def reduced_basis_surrogate(H0, H1, thetas):
    """Offline stage: certified brackets and ground vectors at each
    snapshot theta, plus an orthonormalized snapshot basis B. It also
    returns the exact small Grams B'H0B and B'H1B, which make every
    online Rayleigh quotient computable at k x k cost."""
    from scipy.sparse.linalg import eigsh
    thetas = np.sort(np.asarray(thetas, float))
    vs, los = [], []
    for th in thetas:
        H = H0 + th * H1
        c = eigen_bracket(H)
        los.append(c.value - c.err)
        if len(H) < 64:
            _, V = np.linalg.eigh(H)
            vs.append(V[:, 0])
        else:
            _, V = eigsh(H, k=1, which="SA", v0=_arpack_v0(len(H)))
            vs.append(V[:, 0])
    B, _ = np.linalg.qr(np.column_stack(vs))
    n = len(H0)
    pad = 8 * (n + 2) * np.finfo(float).eps \
        * (float(np.linalg.norm(H0, 1))
           + max(abs(thetas[0]), abs(thetas[-1]))
           * float(np.linalg.norm(H1, 1)))
    return {"thetas": thetas, "lows": np.array(los),
            "A0": B.T @ H0 @ B, "A1": B.T @ H1 @ B, "pad": pad,
            "k": B.shape[1]}


def reduced_basis_bracket(sur, theta: float) -> Certified:
    """Online stage: certified bracket on lambda_min(H0 + theta H1) at
    k x k cost. Upper: exact reduced Rayleigh quotient (variational).
    Lower: the concavity chord through the two bracketing certified
    snapshot lows. Refuses outside the snapshot hull."""
    t = sur["thetas"]
    if theta < t[0] or theta > t[-1]:
        raise ValueError(f"theta={theta:g} outside the snapshot hull "
                         f"[{t[0]:g}, {t[-1]:g}]: concavity certifies "
                         "chords, not extrapolations")
    up = float(np.linalg.eigvalsh(sur["A0"] + theta * sur["A1"])[0]) \
        + sur["pad"]
    j = int(np.searchsorted(t, theta))
    j = max(1, min(j, len(t) - 1))
    w = (theta - t[j - 1]) / (t[j] - t[j - 1])
    lo = (1 - w) * sur["lows"][j - 1] + w * sur["lows"][j] - sur["pad"]
    return Certified(0.5 * (up + lo), 0.5 * (up - lo), Tier.RIGOROUS,
                     (f"rb-ec k={sur['k']} theta={theta:g} "
                      "chord-lower rayleigh-upper",))


# ------------------------------------------------------------- Quantum
# dynamics dispatch (Lieb-Robinson cone). The query with commercial
# teeth: given a local Hamiltonian, a local observable, a time and a
# tolerance, does a classical simulation with a certificate exist, and
# at what cost? The rewrite simulates only a cone of sites around the
# observable and certifies the truncation A-POSTERIORI: by Duhamel,
#   ||A(t) - A_cone(t)|| <= int_0^t ||[H - H_cone, A_cone(s)]|| ds,
# and only the two bonds crossing the cone boundary fail to commute
# with the cone-supported A_cone(s), so the integrand is MEASURED
# inside the cone simulation itself, no Lieb-Robinson velocity
# constants, and the bound is near-zero until the light cone
# physically reaches the edge. Quadrature is made rigorous by the
# Banach-valued interpolation remainder: with K(s) = [P, A_cone(s)],
#   int ||K|| over a cell <= trapezoid + (dt^3/8) sup ||K''||,
# (||T1(s)|| <= interpolated endpoint norms by convexity, and
# ||K - T1|| <= (s-a)(b-s)/2 sup||K''|| from the integral remainder),
# where ||K''|| = ||[P,[H,[H,A]]]|| is MEASURED at the samples and its
# cell drift capped by the crude ||ad_P ad_H^3 A|| <= 16 J ||H||^3,
# harmless behind dt^4. Certificate exact-arithmetic (FP not carried
# in this pipeline).


def _opnorm_ub(M):
    """Cheap rigorous upper bound on the spectral norm:
    min(Frobenius, sqrt(||M||_1 ||M||_inf))."""
    am = np.abs(M)
    return float(min(np.linalg.norm(M),
                     math.sqrt(np.max(am.sum(0)) * np.max(am.sum(1)))))


def _lr_cone_run(n, site, t, J, g, r, n_steps):
    """One cone of the TFI chain H = -J sum ZZ - g sum X. Evolve Z_site
    as a dense operator under the cone Hamiltonian and return
    (value, err). value is <all-up| Z_site(t) |all-up>, and err is the
    certified truncation plus quadrature bound (see the section
    comment). err is exactly 0 when the cone covers the whole chain."""
    lo, hi = max(0, site - r), min(n - 1, site + r)
    nc = hi - lo + 1
    dim = 1 << nc
    bits = (np.arange(dim)[:, None] >> np.arange(nc - 1, -1, -1)[None, :]) & 1
    z = 1.0 - 2.0 * bits                    # z[:, i]: Z eigenvalue, site i
    diag = -J * np.sum(z[:, :-1] * z[:, 1:], axis=1)
    H = np.diag(diag)
    idx = np.arange(dim)
    masks = [1 << (nc - 1 - i) for i in range(nc)]
    for m in masks:
        H[idx, idx ^ m] += -g
    hnorm = J * (nc - 1) + g * nc           # ||H_cone|| by triangle ineq

    def commH(M):
        """[H, M] via the diagonal + bit-flip structure, O(nc dim^2)."""
        HM = diag[:, None] * M
        MH = M * diag[None, :]
        for m in masks:
            p = idx ^ m
            HM -= g * M[p, :]
            MH -= g * M[:, p]
        return HM - MH

    A = np.diag(z[:, site - lo]).astype(complex)
    edges = ([0] if lo > 0 else []) + ([nc - 1] if hi < n - 1 else [])
    if t == 0.0 or not edges:
        if t != 0.0:
            lam, V = np.linalg.eigh(H)
            ph = np.exp(-1j * lam * t)
            A = (V * ph.conj()) @ (V.T @ A @ V * ph[None, :]) @ V.T
        return float(np.real(A[0, 0])), 0.0

    lam, V = np.linalg.eigh(H)
    dt = t / n_steps
    Ud = (V * np.exp(-1j * lam * dt)) @ V.T   # e^{-i H dt}, V real
    gs = {e: np.empty(n_steps + 1) for e in edges}
    Es = {e: np.empty(n_steps + 1) for e in edges}
    for k in range(n_steps + 1):
        C2 = commH(commH(A))
        for e in edges:
            ze = z[:, e]
            gs[e][k] = J * _opnorm_ub(ze[:, None] * A - A * ze[None, :])
            Es[e][k] = J * _opnorm_ub(ze[:, None] * C2 - C2 * ze[None, :])
        if k < n_steps:
            A = Ud.conj().T @ A @ Ud
    err = 0.0
    cap = 16.0 * J * hnorm ** 3              # ||ad_P ad_H^3 A|| crude cap
    for e in edges:
        ge, Ee = gs[e], Es[e]
        err += dt * (np.sum(ge) - 0.5 * (ge[0] + ge[-1]))    # trapezoid
        supE = np.maximum(Ee[:-1], Ee[1:]) + 0.5 * dt * cap
        err += float(np.sum(dt ** 3 / 8.0 * supE))           # quad pad
    return float(np.real(A[0, 0])), _up(err)


def tfi_quench_dispatch(n: int, site: int, t: float, tol: float,
                        g: float = 1.0, J: float = 1.0,
                        n_steps: int = 400,
                        max_dim: int = 4096) -> Certified:
    """Quantum-dynamics dispatch: <Z_site(t)> from the all-up state of
    an n-site transverse-field Ising chain, certified within tol by the
    smallest Lieb-Robinson cone whose measured boundary-commutator
    certificate meets it. Cost depends on the cone, never on n.
    Refuses when no cone within max_dim certifies tol, and the refusal
    carries the measured (radius, err) ladder and the price of the next
    cone."""
    if t < 0:
        raise ValueError("t must be >= 0")

    def dim(r):
        return 1 << (min(n - 1, site + r) - max(0, site - r) + 1)

    rs = []
    r = 1
    while dim(r) <= max_dim:
        rs.append(r)
        if site - r <= 0 and site + r >= n - 1:
            break  # the cone covers the chain; no larger rung exists
        r += 1

    def run(r):
        value, err = _lr_cone_run(n, site, t, J, g, r, n_steps)
        nc = min(n - 1, site + r) - max(0, site - r) + 1
        return Certified(value, err, Tier.RIGOROUS,
                         (f"lr-cone r={r} sites={nc} steps={n_steps} "
                          "a-posteriori boundary commutator, "
                          "exact-arithmetic",))

    def price():
        nd = dim(rs[-1] + 1) if rs else dim(1)
        return (f"the next cone needs dim {nd} (~{(nd / max_dim) ** 3:.0f}x "
                "the largest affordable step cost)")

    rw = Rewrite("lr-cone", tuple(rs), lambda r: float(dim(r)), run, price)
    return plan("lr-dispatch", tol, [rw], jump=False, context=f"t={t:g}")


# ------------------------------------------------------------- Plasma
# hierarchy (guiding-center reduction): the ASYMPTOTIC tier's first
# shipment. The tier's semantics, made concrete: the truncation
# EXPONENT of a reduction is a theorem, guiding-center drift theory
# errs at O(eps^{order+1}) in eps = gyroradius/gradient scale, but
# the CONSTANT is not computable. So the certificate measures it where
# measuring is cheap: the full kinetic solve costs ~1/eps, so a ladder
# of LARGE eps calibrates the constant with cheap solves, the observed
# convergence order is checked against the proven exponent (refusing
# if it falls short, the asymptotic regime has not set in), the most
# pessimistic ladder constant is kept with a declared safety factor,
# and the bound is extrapolated DOWN to the query's small eps, where
# the kinetic solve would be expensive. Proven (the exponent), measured
# (the constant, the observed order), and assumed (no higher-term
# takeover below the ladder) are all named in the provenance, that is
# what Tier.ASYMPTOTIC means. Truth functional: the guiding-center
# transform X = x + (v x zhat)/Omega applied to the true orbit's
# endpoints, operational and gyrophase-free at leading order. Orbit
# solves at rtol 1e-11; solver error not carried (declared), matching
# the Phase 3/4 precedent.


def _gc_orbit_delta(eps, a, v, T):
    """Full kinetic truth: y-displacement of the guiding-center
    transform between the endpoints of the true orbit in
    B = (1 + a x)/eps zhat, unit mass and charge, x0 = 0, v0 = (v, 0)."""
    from scipy.integrate import solve_ivp

    def rhs(t, s):
        B = (1.0 + a * s[0]) / eps
        return (s[2], s[3], s[3] * B, -s[2] * B)

    sol = solve_ivp(rhs, (0.0, T), (0.0, 0.0, v, 0.0), method="DOP853",
                    rtol=1e-11, atol=1e-12)

    def gc_y(s):
        return s[1] - s[2] * eps / (1.0 + a * s[0])   # (v x zhat)_y = -vx

    return float(gc_y(sol.y[:, -1]) - gc_y(sol.y[:, 0]))


@functools.lru_cache(maxsize=None)
def _gc_truth_cached(eps, a, v, T):
    return _gc_orbit_delta(eps, a, v, T)


def _gc_prediction(order, eps, a, v, T):
    """Guiding-center prediction of the same displacement. Order 0:
    no drift. Order 1: grad-B drift v^2 a eps/2 at the initial guiding
    center (b = 1 there; the drift is along y, so b is constant along
    the order-1 trajectory and the integral is closed-form)."""
    if order == 0:
        return 0.0
    if order == 1:
        return eps * v * v * a * T / 2.0
    raise NotImplementedError("guiding-center orders above 1")


def asymptotic_extrapolate(predict: Callable[[float], float],
                           truth: Callable[[float], float],
                           eps: float, k: int, ladder,
                           eta: float = 2.0,
                           ratio_slack: float = 2.0) -> Certified:
    """The ASYMPTOTIC tier's generic certifier. It handles a reduction
    whose truncation exponent k is PROVEN, so error = O(eps^k), but
    whose constant is unknown. Measure the envelope constant
    C = max E(eps)/eps^k on a calibration ladder where truth(eps) is
    affordable. Then certify predict(eps) at any eps at or below the
    ladder floor with err = eta * C * eps^k. The truncation coefficient
    may oscillate, gyrophase-like, so the check is on the ENVELOPE and
    in the one dangerous direction. Refuse when the measured constant
    GROWS toward the ladder floor, either monotonically or because the
    floor rung exceeds ratio_slack times the coarser rungs. That growth
    is the signature of a claimed exponent the data contradict. Refuses
    to extrapolate above the ladder. The provenance names what is proven
    (k), what is measured (C and its spread), and what is assumed (no
    higher-term takeover below the ladder). That declaration is what
    Tier.ASYMPTOTIC means."""
    els = sorted(ladder, reverse=True)
    if eps > els[-1]:
        raise ValueError(f"eps={eps:g} above the calibration ladder floor "
                         f"{els[-1]:g}: refusing to extrapolate upward")
    chat = [abs(predict(el) - truth(el)) / el ** k for el in els]
    grows = all(b > a for a, b in zip(chat, chat[1:])) if len(chat) > 1 \
        else False
    if len(chat) > 1 and (grows or chat[-1] > ratio_slack * max(chat[:-1])):
        raise ValueError(
            "measured constant grows toward the ladder floor "
            f"({', '.join(f'{c:.3g}' for c in chat)}): the claimed order "
            f"{k} is not what the data show")
    C = max(chat)
    spread = C / min(chat) if min(chat) > 0 else math.inf
    return Certified(
        predict(eps), _up(eta * C * eps ** k), Tier.ASYMPTOTIC,
        (f"exponent {k} proven; envelope constant C={C:.3g} measured on "
         f"ladder {tuple(els)} (spread {spread:.1f}x); safety eta={eta:g}; "
         "assumes no higher-term takeover below the ladder",))


def gc_drift_asymptotic(eps: float, order: int = 1, a: float = 0.3,
                        v: float = 1.0, T: float = 25.0,
                        ladder=(0.16, 0.08, 0.04),
                        eta: float = 2.0) -> Certified:
    """ASYMPTOTIC certificate on the order-`order` guiding-center
    prediction at eps, calibrated by full kinetic solves on the (cheap,
    large-eps) ladder. The order-1 truncation coefficient is
    gyrophase-oscillatory here. Measured, its envelope is flat while
    pairwise log-slopes swing wildly. That is exactly why the generic
    certifier checks envelopes rather than slopes."""
    c = asymptotic_extrapolate(
        lambda e: _gc_prediction(order, e, a, v, T),
        lambda e: _gc_truth_cached(e, a, v, T),
        eps, order + 1, ladder, eta)
    return replace(c, provenance=(
        f"gc-hierarchy order={order} eps={eps:g}: " + c.provenance[0],))


def gc_drift_dispatch(eps: float, tol: float, a: float = 0.3,
                      v: float = 1.0, T: float = 25.0,
                      **kw) -> Certified:
    """Certified dispatch along the reduction hierarchy: the cheapest
    guiding-center order whose asymptotic certificate meets tol.
    Refuses when the hierarchy is exhausted, and prices the full
    kinetic fallback."""
    rw = Rewrite(
        "gc-hierarchy", (0, 1), lambda o: float(o + 1),
        lambda o: gc_drift_asymptotic(eps, o, a, v, T, **kw),
        lambda: (f"the full kinetic solve (~{int(40 * T / eps)} RHS "
                 "evaluations, cost ~ T/eps) with a rigorous ODE "
                 "certificate is the remaining rung"))
    return plan("plasma-dispatch", tol, [rw], context=f"eps={eps:g}")


# ------------------------------------------------------------- SOS
# transport bounds (turbulence beachhead). The background-flow method
# in miniature: for dx/dt = f(x) polynomial and a quantity Phi, any
# polynomial V gives sup_x [Phi + grad V . f] >= long-time average of
# Phi on every bounded trajectory (the time average of grad V . f = 
# dV/dt vanishes). Minimizing the sup over V is a sum-of-squares
# program. The project split applies verbatim: the SEARCH for V and
# for a Gram matrix is unrigorous float optimization; the CERTIFICATE
# is exact rational arithmetic, the polynomial identity
# U - Phi - grad V . f = m^T Q m is checked coefficient by coefficient
# over Q (the rationals), and Q >= 0 is proven by rational LDL^T. No
# SDP solver is trusted and no float enters the proof. Boundedness of
# every trajectory (which the theorem needs) is itself an SOS
# certificate: K - delta W - grad W . f >= 0 globally for the
# classical Lorenz Lyapunov function W, so W <= K/delta absorbs.
# Polynomials are dicts {(i, j, k): coeff} over x^i y^j z^k, coeff
# type Fraction (exact path) or float (search path), the same code
# runs both.


def _pmul(p, q):
    out = {}
    for a, ca in p.items():
        for b, cb in q.items():
            m = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
            out[m] = out.get(m, 0 * ca) + ca * cb
    return {m: c for m, c in out.items() if c != 0}


def _paxpy(p, q, s):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0 * c) + s * c
    return {m: c for m, c in out.items() if c != 0}


def _pdiff(p, i):
    out = {}
    for m, c in p.items():
        if m[i] > 0:
            mm = list(m)
            mm[i] -= 1
            out[tuple(mm)] = c * m[i]
    return out


def _lorenz_fields(num=float):
    from fractions import Fraction
    beta = 8.0 / 3.0 if num is float else Fraction(8, 3)
    fx = {(0, 1, 0): 10, (1, 0, 0): -10}
    fy = {(1, 0, 0): 28, (1, 0, 1): -1, (0, 1, 0): -1}
    fz = {(1, 1, 0): 1, (0, 0, 1): -beta}
    return (fx, fy, fz), (10, 28, beta)


def _grad_dot_f(V, fields):
    out = {}
    for i, fi in enumerate(fields):
        out = _paxpy(out, _pmul(_pdiff(V, i), fi), 1)
    return out


def _sos_structure(S):
    """Gram basis for S split by the (x,y) -> (-x,-y) parity when S
    respects it: blocks of monomials of degree <= deg(S)/2, grouped so
    each Gram entry contributes to exactly one monomial of S."""
    deg = max((sum(m) for m in S), default=0)
    half = (deg + 1) // 2
    # per-variable cap floor(deg_v(S)/2) is sound: the v-leading
    # coefficient of sum p_k^2 is itself a nonzero SOS, so no square
    # may exceed half the v-degree of S
    vcap = [max((m[v] for m in S), default=0) // 2 for v in range(3)]
    symmetric = all((m[0] + m[1]) % 2 == 0 for m in S)
    basis = []
    for i in range(min(half, vcap[0]) + 1):
        for j in range(min(half - i, vcap[1]) + 1):
            for k in range(min(half - i - j, vcap[2]) + 1):
                basis.append((i, j, k))
    if symmetric:
        blocks = [[m for m in basis if (m[0] + m[1]) % 2 == 0],
                  [m for m in basis if (m[0] + m[1]) % 2 == 1]]
    else:
        blocks = [basis]
    blocks = [b for b in blocks if b]
    entries = []          # (block, i, j, product monomial, weight)
    groups = {}           # monomial -> entry indices
    for bi, b in enumerate(blocks):
        for i in range(len(b)):
            for j in range(i, len(b)):
                m = tuple(b[i][t] + b[j][t] for t in range(3))
                w = 1 if i == j else 2
                groups.setdefault(m, []).append(len(entries))
                entries.append((bi, i, j, m, w))
    return blocks, entries, groups


def _gram_matrices(blocks, entries, vals):
    mats = [np.zeros((len(b), len(b))) for b in blocks]
    for (bi, i, j, _, _), v in zip(entries, vals):
        mats[bi][i, j] = mats[bi][j, i] = v
    return mats


def _sos_search(S_float, blocks, entries, groups, restarts=3, seed=0):
    """Float search for a PSD Gram: free entries per monomial group
    (the last entry absorbs the exact identity), maximize the minimum
    eigenvalue across blocks. Returns (lambda_min, entry values)."""
    from scipy.optimize import minimize
    free = [i for m, idx in groups.items() for i in idx[:-1]]

    def assemble(theta):
        vals = np.zeros(len(entries))
        vals[free] = theta
        for m, idx in groups.items():
            w_last = entries[idx[-1]][4]
            acc = sum(vals[i] * entries[i][4] for i in idx[:-1])
            vals[idx[-1]] = (S_float.get(m, 0.0) - acc) / w_last
        return vals

    def negmin(theta):
        return -min(np.linalg.eigvalsh(M)[0]
                    for M in _gram_matrices(blocks, entries,
                                            assemble(theta)))

    if not free:
        vals = assemble(np.zeros(0))
        return -negmin(np.zeros(0)), vals
    rng = np.random.default_rng(seed)
    best = (-math.inf, None)
    for r in range(restarts):
        x0 = np.zeros(len(free)) if r == 0 \
            else 0.1 * rng.standard_normal(len(free))
        res = minimize(negmin, x0, method="Powell",
                       options={"maxiter": 4000, "xtol": 1e-10})
        if -res.fun > best[0]:
            best = (-res.fun, assemble(res.x))
    return best


def _rational_ldl_psd(M):
    """Exact LDL^T over Fraction: True iff M is PSD (zero pivots must
    annihilate their entire column)."""
    from fractions import Fraction
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
    for j in range(n):
        if A[j][j] < 0:
            return False
        if A[j][j] == 0:
            if any(A[i][j] != 0 for i in range(j + 1, n)):
                return False
            continue
        for i in range(j + 1, n):
            lij = A[i][j] / A[j][j]
            for k in range(j, n):
                A[i][k] -= lij * A[j][k]
    return True


def _sos_exact_check(S, margin=1e-7, structure=None, warm=None):
    """Exact-rational global-nonnegativity certificate for the
    polynomial S (dict of Fraction coefficients). Float-search a Gram
    matrix, or take the caller's candidate via structure/warm. Any
    solver may propose and none is trusted. Then rationalize its free
    entries, restore the polynomial identity EXACTLY through the
    absorber entries, and prove each block PSD by rational LDL^T. The
    result is one-sided. True is a theorem, and False means no
    certificate was found."""
    from fractions import Fraction
    blocks, entries, groups = structure if structure is not None \
        else _sos_structure(S)
    for m, c in S.items():
        if m not in groups and c != 0:
            return False
    if warm is not None:
        vals = warm
        lam = min(np.linalg.eigvalsh(M)[0]
                  for M in _gram_matrices(blocks, entries, vals))
    else:
        lam, vals = _sos_search({m: float(c) for m, c in S.items()},
                                blocks, entries, groups)
    # the exact LDL^T is the arbiter (zero pivots are legal PSD);
    # the float lambda only gates hopeless cases, rationalization of
    # free entries can perturb a boundary Gram, and then LDL refuses,
    # which is the safe direction
    if vals is None or lam < -abs(margin):
        return False
    # boundary Grams (lam ~ 0) are legal but fragile under rounding:
    # retry at coarser denominators, which recover the simple rational
    # structure such optima usually have
    for den in (10 ** 9, 10 ** 6, 10 ** 3, 10, 1):
        exact = [Fraction(0)] * len(entries)
        for m, idx in groups.items():
            for i in idx[:-1]:
                exact[i] = Fraction(vals[i]).limit_denominator(den)
            last = entries[idx[-1]]
            acc = sum(exact[i] * entries[i][4] for i in idx[:-1])
            exact[idx[-1]] = (S.get(m, Fraction(0)) - acc) / last[4]
        ok = True
        for bi, b in enumerate(blocks):
            M = [[Fraction(0)] * len(b) for _ in b]
            for (bj, i, j, _, _), v in zip(entries, exact):
                if bj == bi:
                    M[i][j] = M[j][i] = v
            if not _rational_ldl_psd(M):
                ok = False
                break
        if ok:
            return True
    return False


def _lorenz_absorbing_certificate():
    """K - delta W - grad W . f is globally SOS for the classical
    W = x^2 + y^2 + (z - sigma - rho)^2: every trajectory enters and
    stays in {W <= K/delta}, so time averages of polynomials exist
    bounded and the auxiliary-function theorem applies to ALL
    trajectories."""
    from fractions import Fraction
    fields, (sigma, rho, beta) = _lorenz_fields(num=Fraction)
    s = sigma + rho
    W = {(2, 0, 0): Fraction(1), (0, 2, 0): Fraction(1),
         (0, 0, 2): Fraction(1), (0, 0, 1): -2 * s, (0, 0, 0): s * s}
    delta = Fraction(3, 2)
    S = _paxpy({}, W, -delta)
    S = _paxpy(S, _grad_dot_f(W, fields), -1)
    zc = -S.get((0, 0, 1), Fraction(0))
    zq = S.get((0, 0, 2), Fraction(0))
    K = zc * zc / (4 * zq) - S.get((0, 0, 0), Fraction(0)) + 1
    S = _paxpy(S, {(0, 0, 0): Fraction(1)}, K)
    return _sos_exact_check(S), K / delta


_LORENZ_V4_BASIS = (
    {(0, 0, 1): 1}, {(2, 0, 0): 1}, {(1, 1, 0): 1},
    {(0, 2, 0): 1}, {(0, 0, 2): 1},
    {(2, 0, 1): 1}, {(1, 1, 1): 1}, {(0, 2, 1): 1},
    {(0, 0, 3): 1},
    {(4, 0, 0): 1},
    {(2, 2, 0): 1, (2, 0, 2): 1},                # x^2 (y^2 + z^2)
    {(0, 4, 0): 1, (0, 2, 2): 2, (0, 0, 4): 1},  # (y^2 + z^2)^2
)


def _lorenz_S(theta, U, num=float):
    """S = U - z - grad V . f with V = sum theta_i B_i; the quartic
    basis elements depend on (y, z) only through y^2 + z^2, which
    cancels the degree-5 part of grad V . f exactly."""
    fields, _ = _lorenz_fields(num=num)
    V = {}
    for t, B in zip(theta, _LORENZ_V4_BASIS):
        V = _paxpy(V, B, t)
    S = {(0, 0, 0): U, (0, 0, 1): -1}
    return _paxpy(S, _grad_dot_f(V, fields), -1)


def lorenz_mean_z_bracket(degree: int = 4) -> Certified:
    """Certified bracket on the supremum over trajectories of the
    long-time average of z for the classical Lorenz system
    (sigma, rho, beta) = (10, 28, 8/3). Upper bound: auxiliary
    polynomial V of the given degree with U - z - grad V . f proven
    globally SOS in exact rational arithmetic. Lower bound: the fixed
    points C+- are exact trajectories with <z> = rho - 1 = 27
    (algebraic identity). Boundedness of all trajectories by the
    absorbing-ball SOS certificate. Tier RIGOROUS throughout. The float
    search chose V, and the rational proof never trusted it."""
    from fractions import Fraction
    ok, ball = _lorenz_absorbing_certificate()
    if not ok:
        raise RuntimeError("absorbing-ball certificate failed")
    rho = Fraction(28)
    if degree == 2:
        # hand solution: V = (y^2 + z^2)/(2 rho beta) - z/beta, U = rho
        b = Fraction(3, 448)
        theta = [Fraction(-3, 8), 0, 0, b, b, 0, 0, 0, 0, 0, 0, 0]
        U = rho
        if not _sos_exact_check(_lorenz_S(theta, U, num=Fraction)):
            raise RuntimeError("degree-2 certificate failed")
    elif degree == 4:
        # The search is a genuine SDP; naive coordinate/subgradient
        # ascent stalls on the thin curved feasible sliver (measured:
        # Powell -7e-8, alternating projections -3e-3). cvxpy+SCS is a
        # SEARCH-ONLY dependency; the proof below never trusts it.
        # and the problem must be nondimensionalized (x -> 25 u, exact)
        # or SCS returns garbage (measured: U*=33 raw vs 27.000004
        # scaled).
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError(
                "degree-4 search uses cvxpy (search-only; the exact "
                "rational proof never trusts it): pip install "
                "cvxpy-base scs") from exc
        L = 25
        U = Fraction(27001, 1000)
        nb = len(_LORENZ_V4_BASIS)

        def scaled(p):
            return {m: c * L ** sum(m) for m, c in p.items()}

        base0 = _lorenz_S([0.0] * nb, 0.0, num=float)
        Sj = []
        for i in range(nb):
            e = [0.0] * nb
            e[i] = 1.0
            d = _lorenz_S(e, 0.0, num=float)
            Sj.append(scaled({m: c - base0.get(m, 0.0)
                              for m, c in d.items()
                              if c != base0.get(m, 0.0)}))
        support = set().union(*[set(d) for d in Sj]) \
            | {(0, 0, 0), (0, 0, 1)}
        blocks, entries, groups = _sos_structure(
            dict.fromkeys(support, 1.0))
        th = cp.Variable(nb)
        t = cp.Variable()
        Q = [cp.Variable((len(b), len(b)), symmetric=True)
             for b in blocks]
        cons = [q >> t * np.eye(q.shape[0]) for q in Q]
        for m in sorted(groups):
            lhs = 0
            for i in groups[m]:
                bi, r, cc, _, w = entries[i]
                lhs = lhs + w * Q[bi][r, cc]
            rhs = (float(U) if m == (0, 0, 0) else 0.0) \
                + (-float(L) if m == (0, 0, 1) else 0.0) \
                + sum(Sj[j].get(m, 0.0) * th[j] for j in range(nb))
            cons.append(lhs == rhs)
        prob = cp.Problem(cp.Maximize(t), cons)
        import warnings
        with warnings.catch_warnings():
            # SCS self-reports "inaccurate" near its tolerance floor;
            # irrelevant here, the exact rational check is the arbiter
            warnings.simplefilter("ignore")
            prob.solve(solver=cp.SCS, eps=1e-10, max_iters=400000)
        if t.value is None or t.value < 1e-6:
            raise RuntimeError(f"degree-4 SDP search failed "
                               f"(status {prob.status})")
        # theta enters the scaled polynomial amplified by L^4 ~ 4e5:
        # rationalize far below the Gram margin or the absorber repair
        # eats it (measured: 1e-7 rounding -> 0.04 coefficient shifts)
        theta = [Fraction(v).limit_denominator(10 ** 13)
                 for v in th.value]
        S_ex = {m: c * Fraction(L) ** sum(m)
                for m, c in _lorenz_S(theta, U, num=Fraction).items()}
        warm = np.zeros(len(entries))
        for k, (bi, i, j, _, _w) in enumerate(entries):
            warm[k] = Q[bi].value[i, j]
        if not _sos_exact_check(S_ex, structure=(blocks, entries, groups),
                                warm=warm):
            raise RuntimeError("degree-4 exact certification failed at "
                               f"U={float(U):.6f}")
    else:
        raise NotImplementedError("degrees 2 and 4")
    lo, up = Fraction(27), U
    return Certified(float((lo + up) / 2), float((up - lo) / 2),
                     Tier.RIGOROUS,
                     (f"lorenz-<z> sos degree={degree}: upper {float(up):.6f} "
                      "by exact rational Gram (LDL^T over Q), lower 27 by "
                      "fixed-point witness (algebraic); all trajectories "
                      f"bounded by absorbing-ball sos certificate (W <= "
                      f"{float(ball):.0f}); long-time averages",))


# ------------------------------------------------------------- GW
# surrogates (the last TARGETS domain). The purest resolution-limited
# query in physics: a waveform need only match to the detector's
# noise-weighted mismatch, epsilon set by instrument and SNR, not by
# formalism. The rewrite is the standard surrogate pipeline (greedy
# reduced basis over a waveform manifold + smooth coefficient fits),
# and the certificate is the field's own practice made honest:
# accuracy studies against held-out truth, upgraded to a
# distribution-free conformal guarantee, the worst mismatch over
# n_cal held-out parameter draws bounds a fresh draw from the SAME
# distribution with P(miss) <= 1/(n_cal+1), by exchangeability alone.
# Tier EMPIRICAL with the fail probability printed; refusal outside
# the training hull and below the calibrated mismatch. The truth
# family here is a declared Newtonian-chirp-shaped model (the demo's
# stand-in for numerical relativity); nothing in the pipeline peeks
# at closed forms, only waveform evaluations, as with real NR data.


_GW_T = np.linspace(0.0, 1.0, 4096, endpoint=False)


def _gw_chirp(lam: float) -> np.ndarray:
    """Model waveform family, complex analytic signal: a Newtonian
    chirp phase plus a PN-flavored correction with a different lam
    power, amplitude rising to merger with its own lam dependence.
    lam plays chirp mass, giving ~40 cycles at lam=1 down to ~26 at
    lam=2, so raw waveforms decorrelate strongly across the range. The
    surrogate must discover the smooth amplitude and phase structure
    from data, exactly as with numerical-relativity input."""
    tau = 1.02 - _GW_T
    phase = -250.0 * (tau / lam) ** 0.625 - 40.0 * tau ** 0.375 / lam ** 1.125
    amp = tau ** -0.25 * (1.0 + 0.15 * np.sqrt(tau) / lam)
    return amp * np.exp(1j * phase)


def _gw_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    """1 - |<a,b>| / (||a|| ||b||): the |.| maximizes the overlap over
    a global phase (no time marginalization, declared)."""
    return float(1.0 - abs(np.vdot(a, b))
                 / (np.linalg.norm(a) * np.linalg.norm(b)))


def gw_surrogate_build(lam_range=(1.0, 2.0), n_train: int = 48,
                       eps_build: float = 1e-8, n_cal: int = 49,
                       fit_degree: int = 14, seed: int = 0) -> dict:
    """Offline stage, the gwsurrogate architecture in miniature. Raw
    waveforms decorrelate wildly in lam, with hundreds of radians of
    dephasing, so no basis fits them directly. But amplitude and
    unwrapped phase are SMOOTH in lam. Fix the free global phase by
    rotating each training waveform to h(0) real-positive, which
    mismatch maximizes over anyway. Unwrap the phase along t. Then SVD
    each of the amplitude and phase matrices, keeping modes while the
    relative singular-value tail exceeds eps_build, and Chebyshev-fit
    the mode coefficients over lam. Conformal calibration on n_cal fresh
    uniform draws records the worst mismatch m_cal. All expensive truth
    evaluations happen offline, never online."""
    lo, hi = lam_range
    lams = np.linspace(lo, hi, n_train)
    H = np.stack([_gw_chirp(l) for l in lams])
    H *= np.exp(-1j * np.angle(H[:, 0]))[:, None]
    A = np.abs(H)
    Phi = np.unwrap(np.angle(H), axis=1)
    from numpy.polynomial import chebyshev
    x = (2 * lams - lo - hi) / (hi - lo)
    deg = min(fit_degree, n_train - 1)

    def svd_fit(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        k = max(1, int(np.sum(s > eps_build * s[0])))
        return [(chebyshev.chebfit(x, U[:, i] * s[i], deg), Vt[i])
                for i in range(k)]

    sur = {"range": (lo, hi), "amp": svd_fit(A), "phase": svd_fit(Phi),
           "n_cal": n_cal}
    sur["basis"] = [v for _, v in sur["amp"]] \
        + [v for _, v in sur["phase"]]
    rng = np.random.default_rng(seed)
    m_cal = 0.0
    for lam in rng.uniform(lo, hi, n_cal):
        m = _gw_mismatch(_gw_surrogate_raw(sur, float(lam)),
                         _gw_chirp(float(lam)))
        m_cal = max(m_cal, m)
    sur["m_cal"] = m_cal
    return sur


def _gw_surrogate_raw(sur, lam):
    lo, hi = sur["range"]
    from numpy.polynomial import chebyshev
    x = (2 * lam - lo - hi) / (hi - lo)
    amp = sum(chebyshev.chebval(x, c) * v for c, v in sur["amp"])
    phi = sum(chebyshev.chebval(x, c) * v for c, v in sur["phase"])
    h = amp * np.exp(1j * phi)
    return h / np.linalg.norm(h)


def gw_surrogate_eval(sur, lam: float) -> Certified:
    """Online query: the surrogate waveform at lam with the conformal
    mismatch certificate. err is the 2-norm bound sqrt(2 m_cal) up to
    a global phase (min over phase of ||h_sur - e^{i theta} h_true||),
    valid for lam drawn from the calibration distribution. That is
    declared, like every EMPIRICAL certificate."""
    lo, hi = sur["range"]
    if not lo <= lam <= hi:
        raise ValueError(f"lam={lam:g} outside the training hull "
                         f"[{lo:g}, {hi:g}]")
    h = _gw_surrogate_raw(sur, lam)
    return Certified(
        h, math.sqrt(2.0 * sur["m_cal"]), Tier.EMPIRICAL,
        (f"gw-surrogate n_basis={len(sur['basis'])} lam={lam:g} "
         f"mismatch<={sur['m_cal']:.3g} conformal n_cal={sur['n_cal']} "
         "uniform draws, no time-marginalization",),
        fail_p=1.0 / (sur["n_cal"] + 1))


def gw_surrogate_dispatch(sur, lam: float, tol: float) -> Certified:
    """The crisp-epsilon dispatch: serve the query iff the calibrated
    mismatch meets the detector's tolerance, else refuse with the
    price of improvement."""
    if sur["m_cal"] > tol:
        raise ValueError(
            f"gw-dispatch: calibrated mismatch {sur['m_cal']:.3g} > "
            f"tol={tol:g}; rebuild with tighter eps_build / more "
            f"training points (currently {len(sur['basis'])} basis "
            f"vectors from eps_build offline tolerance)")
    return gw_surrogate_eval(sur, lam)


def gci_extrapolate(vals, hs, safety: float = 3.0, p_floor: float = 0.5,
                    p_spread: float = 0.8, p_cap: float = 2.0) -> Certified:
    """Grid-convergence certificate for a resolution ladder, in the
    manner of Roache's GCI (the standard of engineering solution
    verification). vals are the functional at resolutions hs, coarse
    to fine. The convergence order is MEASURED from ladder triplets,
    so the tier is EMPIRICAL: unlike asymptotic_extrapolate, no
    exponent is proven here. Refuses when the ladder shows no
    asymptotic range: differences that change sign, a measured order
    below p_floor, or orders that disagree across triplets by more
    than p_spread. The certified value is the finest rung; err covers
    the remaining distance to h -> 0 with the declared safety factor."""
    if len(vals) < 3 or len(vals) != len(hs):
        raise ValueError("need >= 3 ladder rungs")
    if any(b >= a for a, b in zip(hs, hs[1:])):
        raise ValueError("hs must decrease, coarse to fine")
    d = [b - a for a, b in zip(vals, vals[1:])]
    if all(x == 0 for x in d):
        return Certified(vals[-1], 0.0, Tier.EMPIRICAL,
                         (f"gci rungs={len(vals)} ladder constant at every "
                          "rung; assumes the constancy persists to h=0",))
    if any(x == 0 for x in d) or any(x * y <= 0 for x, y in zip(d, d[1:])):
        raise ValueError("no asymptotic range: ladder differences are "
                         f"not monotone ({', '.join(f'{x:.3g}' for x in d)})")
    ratios = [a / b for a, b in zip(hs, hs[1:])]
    if max(ratios) / min(ratios) > 1.01:
        raise ValueError("ladder must use a fixed refinement ratio")
    ps = [math.log(abs(d[k] / d[k + 1])) / math.log(ratios[k])
          for k in range(len(d) - 1)]
    if min(ps) < p_floor or max(ps) - min(ps) > p_spread:
        raise ValueError("no asymptotic range: measured orders "
                         f"{', '.join(f'{p:.2f}' for p in ps)} are unstable "
                         f"or below {p_floor:g}")
    # cap the usable order at the scheme's formal order: a coincidentally
    # clean triplet can measure a spurious high order, and the inflated
    # denominator then yields a false, too-tight certificate (measured:
    # p=5.1 on a wall-force peak whose finer rung fell outside the err)
    p = min(min(ps), p_cap)
    r_last = hs[-2] / hs[-1]
    err = safety * abs(d[-1]) / (r_last ** p - 1.0)
    return Certified(vals[-1], _up(err), Tier.EMPIRICAL,
                     (f"gci rungs={len(vals)} order measured "
                      f"p={p:.2f} (not proven) safety={safety:g}; assumes "
                      "the asymptotic range seen on the ladder persists "
                      "to h=0",))


def continuum_limit(rungs, hs, label: str, scale: float = 1.0,
                    unit: str = "") -> Certified:
    """Turn a ladder of certificates about discretized models into one
    certificate about the thing being modelled. Every rung is rigorous
    about its own mesh and silent about the mesh itself. This adds the
    only statement that crosses that line, and it is a measured one.

    Two errors compose, and they are of different kinds. The distance
    from the finest rung to h -> 0 is read off the ladder by
    gci_extrapolate, so it is EMPIRICAL however rigorous the rungs
    were. The finest rung's own error is whatever it was, and it simply
    adds. The tier of the pair is the weaker one, which means a
    continuum answer is never RIGOROUS no matter what it is built from.
    Saying so is the point. The exchange rate between a proven statement
    about a model and a measured one about the world is usually a factor
    of hundreds and a tier. Printing both is what lets a reader see
    it.

    One precondition is checked rather than hoped for. A measured
    convergence order is a ratio of differences between rungs. So if the
    rungs' own error bars are comparable to those differences, the order
    is fitted to bracket noise and the certificate that follows is
    fiction. The rule here is a factor of ten. Falling short of it is a
    refusal that prices the fix: tighten the rungs or coarsen the
    ladder.

    scale and unit only make the refusal readable in the field's own
    units, 10^5 and pcm for a reactor, 1 and nC/cm^2 for a junction.
    They touch no bound."""
    if len(rungs) < 3:
        raise ValueError("a grid-convergence certificate needs >= 3 rungs")
    vals = [c.value for c in rungs]
    diffs = [abs(b - a) for a, b in zip(vals, vals[1:])]
    worst = max(c.err for c in rungs)
    if min(diffs) <= 10.0 * worst:
        raise ValueError(
            f"the ladder cannot be read: the widest discrete bracket "
            f"({worst * scale:.3g}{unit}) is not small against the closest "
            f"pair of rungs ({min(diffs) * scale:.3g}{unit}), so a measured "
            f"convergence order would be fitting bracket noise; tighten "
            f"the per-rung tolerance below "
            f"{min(diffs) * scale / 10:.2g}{unit} or coarsen the ladder")
    g = gci_extrapolate(vals, hs)
    fine = rungs[-1]
    return Certified(g.value, _up(g.err + fine.err), min(g.tier, fine.tier),
                     fine.provenance + g.provenance
                     + (f"{label}, grid error {g.err * scale:.3g}{unit} + "
                        f"finest bracket {fine.err * scale:.3g}{unit}",),
                     min(1.0, g.fail_p + fine.fail_p))


# ------------------------------------------------------------- SPH
# beachhead: wave impact on a wall. The model, declared: 2D weakly
# compressible SPH (Wendland C2 kernel, Tait equation of state,
# Monaghan artificial viscosity, dynamic boundary particles), a
# dam-break column collapsing into a bore that strikes the far wall.
# The engineering question is what the impact does to the wall. The
# query has to be designed before it can be certified: the raw peak
# pressure of a breaking-wave impact is famously irreproducible (it
# depends on entrapped air and the last millimeter of breaker shape),
# and the resolution ladder shows it, no asymptotic range, so the
# GCI certifier refuses. The time-smoothed impact force converges,
# and that is the quantity sea-wall design practice actually uses.
# Certificates here are EMPIRICAL (measured convergence order, GCI):
# for violent free-surface flow no proven exponent exists, and the
# honest tier says so.


def _sph_pairs(px, py, rad):
    """All particle pairs closer than rad, via a cell-linked list.
    Returns (i, j) index arrays with i < j."""
    cx = np.floor(px / rad).astype(np.int64)
    cy = np.floor(py / rad).astype(np.int64)
    ncy = cy.max() - cy.min() + 3
    key = (cx - cx.min() + 1) * ncy + (cy - cy.min() + 1)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    ii, jj = [], []
    for dk in (0, ncy - 1, ncy, ncy + 1, 1):
        tgt = ks + dk
        lo = np.searchsorted(ks, tgt, side="left")
        hi = np.searchsorted(ks, tgt, side="right")
        n = hi - lo
        src = np.repeat(np.arange(len(ks)), n)
        if len(src) == 0:
            continue
        dst = np.repeat(lo, n) + (np.arange(len(src))
                                  - np.repeat(np.cumsum(n) - n, n))
        if dk == 0:
            keep = dst > src
            src, dst = src[keep], dst[keep]
        ii.append(order[src])
        jj.append(order[dst])
    i = np.concatenate(ii)
    j = np.concatenate(jj)
    d2 = (px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2
    keep = d2 < rad * rad
    return i[keep], j[keep]


def sph_dam_break(nres: int = 18, tank=(4.0, 3.0), column=(1.0, 1.0),
                  T: float = 4.6, alpha: float = None,
                  obstacle=None, snapshots=()):
    """Run the dam break and return the horizontal force history on
    the right wall. nres is particles per unit length; g = rho0 = 1;
    the bore reaches the wall near t ~ 2.2 for the default geometry.
    obstacle, if given, is (x0, width, height): a rectangular berm on
    the floor in the bore's path, built from the same boundary
    particles as the walls. Returns a dict with ts, F (wall force),
    and particle snapshots at the requested times."""
    if alpha is None:
        # Monaghan viscosity scales with h, so fixed alpha would give
        # each ladder rung a different fluid (measured: arrival times
        # drifting 3.5 -> 2.8 down a ladder). Scale alpha ~ 1/nres to
        # hold the physical viscosity constant across resolutions.
        alpha = 1.44 / nres
    dx = 1.0 / nres
    h = 1.3 * dx
    rad = 2.0 * h
    rho0, g, gamma = 1.0, 1.0, 7.0
    c0 = 20.0 * math.sqrt(g * column[1])
    B = c0 * c0 * rho0 / gamma
    m = rho0 * dx * dx

    def grid(x0, x1, y0, y1):
        gx = np.arange(x0 + dx / 2, x1, dx)
        gy = np.arange(y0 + dx / 2, y1, dx)
        X, Y = np.meshgrid(gx, gy)
        return X.ravel(), Y.ravel()

    fx, fy = grid(0.0, column[0], 0.0, column[1])
    walls = [grid(-3 * dx, tank[0] + 3 * dx, -3 * dx, 0.0),      # floor
             grid(-3 * dx, 0.0, 0.0, tank[1]),                   # left
             grid(tank[0], tank[0] + 3 * dx, 0.0, tank[1])]      # right
    if obstacle is not None:
        ox, ow, oh = obstacle
        walls.append(grid(ox, ox + ow, 0.0, oh))
    bx = np.concatenate([w[0] for w in walls])
    by = np.concatenate([w[1] for w in walls])
    nf = len(fx)
    right = np.zeros(nf + len(bx), bool)
    right[nf + sum(len(w[0]) for w in walls[:2]):
          nf + sum(len(w[0]) for w in walls[:3])] = True

    px = np.concatenate([fx, bx])
    py = np.concatenate([fy, by])
    vx = np.zeros_like(px)
    vy = np.zeros_like(px)
    rho = np.full(len(px), rho0)
    fluid = np.zeros(len(px), bool)
    fluid[:nf] = True

    dt = 0.25 * h / c0
    nsteps = int(T / dt)
    ts = np.arange(nsteps) * dt
    F = np.zeros(nsteps)
    snaps, want = [], sorted(snapshots)

    a2 = 7.0 / (4.0 * math.pi * h * h)

    def rates():
        i, j = _sph_pairs(px, py, rad)
        dxij = px[i] - px[j]
        dyij = py[i] - py[j]
        r = np.sqrt(dxij ** 2 + dyij ** 2)
        q = r / h
        f1 = np.maximum(1.0 - 0.5 * q, 0.0)
        W = a2 * f1 ** 4 * (1.0 + 2.0 * q)
        gfac = -5.0 * a2 * f1 ** 3 / (h * h)     # grad W = gfac * (dx, dy)
        gx_, gy_ = gfac * dxij, gfac * dyij
        # density by summation every step: drift-free, unlike the
        # continuity form (measured: continuity drifted the hydrostatic
        # pressure to 1.9x at one resolution and 0 at another)
        rho = np.full(len(px), m * a2)           # self term W(0)
        np.add.at(rho, i, m * W)
        np.add.at(rho, j, m * W)
        # free-surface WCSPH: kernel deficiency at the surface reads
        # rho < rho0, and the Tait EOS would turn that into strong
        # negative (tensile) pressure and blow the flow apart; clamp
        p = np.maximum(B * ((rho / rho0) ** gamma - 1.0), 0.0)
        dvx = vx[i] - vx[j]
        dvy = vy[i] - vy[j]
        vr = dvx * dxij + dvy * dyij
        mu = np.where(vr < 0, h * vr / (r * r + 0.01 * h * h), 0.0)
        pi_ij = -alpha * c0 * mu / (0.5 * (rho[i] + rho[j]))
        fac = m * (p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2 + pi_ij)
        ax = np.zeros(len(px))
        ay = np.zeros(len(px))
        np.add.at(ax, i, -fac * gx_)
        np.add.at(ay, i, -fac * gy_)
        np.add.at(ax, j, fac * gx_)
        np.add.at(ay, j, fac * gy_)
        ay[fluid] -= g
        wallF = float(np.sum(np.where(right[j] & fluid[i], m * fac * gx_, 0.0))
                      - np.sum(np.where(right[i] & fluid[j], m * fac * gx_,
                                        0.0)))
        return rho, ax, ay, wallF

    for k in range(nsteps):
        rho, ax, ay, wallF = rates()
        F[k] = wallF
        vx[fluid] += dt * ax[fluid]
        vy[fluid] += dt * ay[fluid]
        px[fluid] += dt * vx[fluid]
        py[fluid] += dt * vy[fluid]
        if want and k * dt >= want[0]:
            pfl = B * ((rho[fluid] / rho0) ** gamma - 1.0)
            snaps.append((want.pop(0), px[fluid].copy(), py[fluid].copy(),
                          pfl))
    return {"ts": ts, "F": F, "snaps": snaps, "n_fluid": nf, "dx": dx,
            "px": px, "py": py, "rho": rho, "fluid": fluid, "B": B,
            "gamma": gamma}


def wave_impact_force(F, ts, tau: float):
    """The queried functional: the largest boxcar average of the wall
    force over a window tau. tau = 0 asks for the raw instantaneous
    peak, the query the ladder refuses to certify."""
    if tau <= 0.0:
        return float(np.max(F))
    w = max(1, int(round(tau / (ts[1] - ts[0]))))
    c = np.cumsum(np.concatenate([[0.0], F]))
    return float(np.max((c[w:] - c[:-w]) / w))


def sph_wall_impulse(nres: int, obstacle=None, T: float = 3.2) -> float:
    """The queried functional: total horizontal impulse delivered to
    the right wall over the first T time units (momentum transferred
    by the impact). Robust where the raw peak is not: it integrates
    out both the pressure-spike scatter and the arrival-time jitter
    between resolutions."""
    out = sph_dam_break(nres=nres, T=T, obstacle=obstacle)
    return float(np.sum(out["F"]) * (out["ts"][1] - out["ts"][0]))


# ------------------------------------------------------------- FEniCSx
# bridge: Grad-Shafranov equilibrium with a guaranteed a posteriori
# certificate. The tokamak equilibrium equation -Delta* psi =
# R^2 p'(psi) + F F'(psi) is weighted Poisson, kappa = 1/R, on the
# (R, Z) half-plane, and for elliptic problems the Prager-Synge
# identity gives error bounds that are GUARANTEED, not estimated:
# for the weak solution u, any conforming u_h with the right boundary
# trace, and ANY H(div) field sigma,
#   |||u - u_h||| <= ||kappa grad u_h + sigma||_{1/kappa}
#                    + sqrt(Rmax/lam1) ||f - div sigma||,
# where lam1 is the exact first Dirichlet eigenvalue of the rectangle
# and |||.||| the kappa-weighted energy norm. FEniCSx proposes: it
# solves the primal problem and a mixed RT problem for sigma. The
# bound holds for whatever it returns. A bad sigma loosens the bound
# and cannot invalidate it (measured: a sign error in the mixed
# boundary term gave efficiency 27x, still a valid bound; fixed,
# 1.6x). Implicit
# coupling c*psi in the source is certified through the contraction
# factor theta = c Rmax / (Rmin lam1), refusing at theta >= 1. Tier
# RIGOROUS; assembly and linear-solver arithmetic are not carried
# (declared), as in the other solver pipelines. The Solov'ev
# polynomial equilibrium supplies an exact solution for the tests.


def legendre_source_profile(A: float, rho: float, k: int) -> Certified:
    """A declared current-profile correction for the Grad-Shafranov
    source: g(R) = sum over j of A rho^j P_j((R - R0)/W), Legendre
    polynomials on the GS rectangle (R0, W, H = 3, 1, 1). The model
    is the full series; a computation can only afford its first k
    terms. Orthogonality prices the truncation exactly: the dropped
    tail has L2(Omega) norm squared sum_{j>=k} A^2 rho^{2j} * 2HW *
    2/(2j+1) <= 2HW * (2/(2k+1)) * A^2 rho^{2k} / (1 - rho^2), a
    rigorous bound in exact arithmetic. The value is the coefficient
    vector, and the err is the tail bound. That is the entry ticket for
    a composed plan that feeds this profile into the equilibrium solve
    and converts this err through the solve's exported
    sensitivity."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1) for a summable tail")
    W, H = 1.0, 1.0
    coeffs = np.array([A * rho ** j for j in range(k)])
    tail = _up(math.sqrt(2 * H * W * (2.0 / (2 * k + 1))
                         * (A * rho ** k) ** 2 / (1 - rho * rho)))
    return Certified(coeffs, tail, Tier.RIGOROUS,
                     (f"legendre-profile k={k} A={A:g} rho={rho:g} "
                      "orthogonal tail bound",))


def _gs_solve(n, c, R0, W, H, a_c, b_c, d_c, degree, dg0=0.0,
              source_coeffs=()):
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    import basix.ufl
    msh = dmesh.create_rectangle(MPI.COMM_WORLD,
                                 [[R0 - W, -H], [R0 + W, H]], [n, n],
                                 dmesh.CellType.triangle)
    x = ufl.SpatialCoordinate(msh)
    R = x[0]
    Z = x[1]
    psi_ex = a_c * (R ** 2 - R0 ** 2) ** 2 + b_c * R ** 2 * Z ** 2 \
        + d_c * Z ** 2
    g0 = -((8 * a_c + 2 * b_c) * R ** 2 + 2 * d_c) + dg0  # -Delta* psi_ex
    if len(source_coeffs):
        # Legendre recurrence built symbolically in UFL:
        # (j+1) P_{j+1} = (2j+1) x P_j - j P_{j-1}
        xhat = (R - R0) / W
        p0, p1 = 1.0, xhat
        gp = source_coeffs[0] * p0
        if len(source_coeffs) > 1:
            gp = gp + source_coeffs[1] * p1
        for j in range(1, len(source_coeffs) - 1):
            p0, p1 = p1, ((2 * j + 1) * xhat * p1 - j * p0) / (j + 1)
            gp = gp + source_coeffs[j + 1] * p1
        g0 = g0 + gp
    kappa = 1.0 / R

    V = fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    aform = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    uD = fem.Function(V)
    uD.interpolate(fem.Expression(psi_ex, V.element.interpolation_points))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = fem.locate_dofs_topological(
        V, tdim - 1, dmesh.exterior_facet_indices(msh.topology))
    bc = fem.dirichletbc(uD, bdofs)

    uh = fem.Function(V)
    uh.interpolate(uD)
    picard = 1 if c == 0.0 else 40
    for _ in range(picard):
        f = (g0 + c * uh) / R
        prob = LinearProblem(
            aform, f * v * ufl.dx, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "ksp_error_if_not_converged": True},
            petsc_options_prefix="gs_")
        sol = prob.solve()
        uh = sol[0] if isinstance(sol, tuple) else sol

    # the frozen source the certificate is issued against
    f = (g0 + c * uh) / R
    RT = basix.ufl.element("RT", msh.basix_cell(), degree)
    DG = basix.ufl.element("DG", msh.basix_cell(), degree - 1)
    Wsp = fem.functionspace(msh, basix.ufl.mixed_element([RT, DG]))
    sig, p = ufl.TrialFunctions(Wsp)
    tau, q = ufl.TestFunctions(Wsp)
    am = (ufl.inner(sig / kappa, tau) - p * ufl.div(tau)
          + ufl.div(sig) * q) * ufl.dx
    Lm = f * q * ufl.dx \
        - psi_ex * ufl.inner(tau, ufl.FacetNormal(msh)) * ufl.ds
    pm = LinearProblem(am, Lm, bcs=[],
                       petsc_options={"ksp_type": "preonly",
                                      "pc_type": "lu",
                                      "pc_factor_mat_solver_type": "mumps",
                                      "ksp_error_if_not_converged": True},
                       petsc_options_prefix="gsm_")
    sol = pm.solve()
    wh = sol[0] if isinstance(sol, tuple) else sol
    sigh = wh.sub(0).collapse()

    def norm(form):
        return math.sqrt(abs(fem.assemble_scalar(fem.form(form))))

    flux = norm(ufl.inner(kappa * ufl.grad(uh) + sigh,
                          kappa * ufl.grad(uh) + sigh) / kappa * ufl.dx)
    osc = norm((f - ufl.div(sigh)) ** 2 * ufl.dx)
    err_meas = norm(kappa * ufl.inner(ufl.grad(uh) - ufl.grad(psi_ex),
                                      ufl.grad(uh) - ufl.grad(psi_ex))
                    * ufl.dx)
    Qh = fem.assemble_scalar(fem.form(uh * ufl.dx))
    return flux, osc, err_meas, float(Qh), uh, msh


def gs_equilibrium_certified(n: int = 16, c: float = 0.0,
                             degree: int = 1, dg0: float = 0.0,
                             source_coeffs=()) -> dict:
    """Fixed-boundary Grad-Shafranov on the rectangle
    [R0-W, R0+W] x [-H, H] with the Solov'ev source plus an implicit
    coupling c*psi, solved by FEniCSx (untrusted) and certified by the
    Prager-Synge bound with rectangle-exact constants. Returns the
    energy-norm bound and a Certified value of the total poloidal
    flux content, integral of psi over the domain. Refuses when the
    coupling exceeds the contraction limit. dg0 adds a constant to the
    source profile, which is a uniform current-density offset.
    source_coeffs adds a Legendre profile in (R - R0)/W, the shape
    legendre_source_profile certifies.

    The flux also exports its sensitivity to that source. Subtract the
    weak forms of two coupled solutions with the same boundary data.
    The difference e then obeys (1-theta)|||e|||^2 <=
    sqrt(Rmax/lam1) ||dg||/Rmin |||e|||, and |Q moves| <=
    sqrt(area Rmax/lam1) |||e|||. Chained, |Q moves| <=
    sqrt(area) Rmax / (lam1 Rmin (1-theta)) * ||dg||_L2. That is the
    same contraction machinery that certifies the solve, repriced as a
    Lipschitz bound. It is the datum a composed plan needs before
    feeding this solve from an upstream certified profile."""
    R0, W, H = 3.0, 1.0, 1.0
    # O-point (magnetic axis) at (R0, 0): needs 9 b + d > 0;
    # these give psi_RR/psi_ZZ ~ 2, a mildly elongated core
    a_c, b_c, d_c = 1.0 / 100, 1.0 / 108, 1.0 / 10
    Rmin, Rmax = R0 - W, R0 + W
    lam1 = math.pi ** 2 * (1.0 / (2 * W) ** 2 + 1.0 / (2 * H) ** 2)
    theta = c * Rmax / (Rmin * lam1)
    if theta >= 0.95:
        raise ValueError(
            f"contraction factor theta={theta:.2f} >= 0.95: the coupled "
            f"source c*psi is not certifiably contractive on this domain "
            f"(limit c < {0.95 * Rmin * lam1 / Rmax:.2f})")
    flux, osc, err_meas, Qh, uh, msh = _gs_solve(n, c, R0, W, H, a_c, b_c,
                                                 d_c, degree, dg0,
                                                 source_coeffs)
    eta = flux + math.sqrt(Rmax / lam1) * osc
    energy_bound = _up(eta / (1.0 - theta))
    # |Q(u) - Q(u_h)| <= ||1|| ||u - u_h|| <= sqrt(|Omega| Rmax/lam1) |||e|||
    area = 4.0 * W * H
    q_err = _up(math.sqrt(area * Rmax / lam1) * energy_bound)
    sens = Sensitivity(_up(math.sqrt(area) * Rmax
                           / (lam1 * Rmin * (1.0 - theta))),
                       Tier.RIGOROUS, "source")
    Q = Certified(Qh, q_err, Tier.RIGOROUS,
                  (f"gs-equilibrium n={n} c={c:g}"
                   + (f" dg0={dg0:g}" if dg0 else "")
                   + (f" profile-k={len(source_coeffs)}"
                      if len(source_coeffs) else "")
                   + f" prager-synge flux+osc "
                   f"({flux:.3g}+{osc:.3g}) rectangle-exact lam1, "
                   f"contraction theta={theta:.2f}; assembly and solver "
                   "arithmetic not carried",),
                  sensitivity=sens)
    return {"Q": Q, "energy_bound": energy_bound, "err_measured": err_meas,
            "flux_term": flux, "osc_term": osc, "theta": theta,
            "uh": uh, "msh": msh}


# ------------------------------------------------- reactor criticality.
# Every eigenvalue bracket above this line runs on the variational
# theorem, which needs a self-adjoint operator: lambda_min is a minimum,
# so any trial state overshoots it, and that is the whole proof. A
# reactor is not self-adjoint. Neutrons scatter DOWN in energy and
# almost never up, so the two-group loss operator is block
# lower-triangular and its transpose is a different physical problem
# (the adjoint flux, importance). The minimum principle certifies
# nothing here, and asking LAPACK for the eigenvalue does not help
# either: a computed eigenvalue of a non-symmetric matrix has no cheap
# rigorous bound, because its condition number is the reciprocal of an
# eigenvector overlap that can be arbitrarily small.
#
# What survives is POSITIVITY. Neutrons cannot be negative, and every
# operator in the problem respects that: fission makes neutrons
# (F >= 0), and the loss operator inverts to a non-negative Green's
# function (L^-1 >= 0) because a source anywhere produces flux
# everywhere and nowhere a deficit. An operator that maps the
# non-negative cone into itself has a Perron-Frobenius eigenvalue, and
# Collatz and Wielandt showed how to sandwich it: for A >= 0 and ANY
# strictly positive x,
#     min_i (Ax)_i / x_i  <=  rho(A)  <=  max_i (Ax)_i / x_i,
# because Ax >= m x implies A^k x >= m^k x and likewise from above.
# Here A = L^-1 F and rho(A) is k_eff, the number reactor physics is
# organized around. The bracket costs one application of A, which is
# also exactly one step of the fission-source iteration every reactor
# code already runs. Production codes stop that iteration when k stops
# moving, which is a hope; this stops when the sandwich closes, which
# is a bound.


def _fl_matvec(M, x):
    """A matrix-vector product and a rigorous bound on its own rounding
    error, componentwise. The classical dot-product bound says a sum of
    m products computed in floating point lands within gamma_m times the
    sum of the absolute terms, gamma_m = m*eps/(1 - m*eps), so one extra
    product of absolute values buys the certificate. Cheaper than
    carrying the Interval class through a 200x200 matrix, and the same
    trick the reduced-basis pad uses.

    The factor two is not decoration. The bound is itself computed in
    floating point -- gamma_m rounds, and so does the sum of absolute
    terms -- so a bound quoted exactly at gamma_m could fall a few ulps
    short of what it claims. Doubling swallows every such second-order
    term at a cost of one bit, which on this problem is a million times
    below anything that matters."""
    y = M @ x
    m = int(max(1, (M != 0).sum(1).max()))
    return y, 2.0 * m * np.finfo(float).eps * (np.abs(M) @ np.abs(x))


def mmatrix_witness(L: np.ndarray) -> np.ndarray:
    """Proof that L inverts to something non-negative, which is what the
    whole certificate rests on. Two facts do it. First, L must be a
    Z-matrix, meaning every off-diagonal entry is non-positive, checked
    exactly on the stored floats. Second, there must exist a strictly
    positive u with L u >= 1 everywhere. A Z-matrix admitting one is a
    nonsingular M-matrix, so L^-1 >= 0, and the same u bounds the
    inverse, since L^-1 e <= u componentwise.

    We find u by solving L u = 1 and scaling up for slack, then VERIFY
    the result against a bound on the check's own rounding. How u was
    found does not matter. What is checked is that L u >= 1 holds, and
    a solver that lied would fail the check rather than corrupt the
    bound. That is the proposer and checker split the whole library
    runs on, applied to a hypothesis instead of an answer. Both tests
    are needed. Flipping one off-diagonal of a reactor operator positive
    leaves a u that still passes the second check, and only the
    Z-matrix test catches it.

    The scaling is measured, not guessed. A fixed nudge is a bet that
    the solve came back accurate to better than the nudge, and the bet
    is lost the moment L is ill-conditioned. The Jacobian of a
    semiconductor's nonlinear Poisson equation, written the way the
    physics writes it, has a condition number in the hundreds of
    millions. The solve is then wrong in the eighth digit, and a 10^-9
    nudge leaves L u short of 1. Scaling u by its own measured
    shortfall costs one extra matvec, needs no constant, and cannot be
    outgrown, because both the shortfall and the rounding pad scale
    with u."""
    off = L - np.diag(np.diag(L))
    if (off > 0).any():
        i, j = np.unravel_index(np.argmax(off), off.shape)
        raise ValueError(
            f"not a Z-matrix: L[{i},{j}] = {off[i, j]:g} > 0, so "
            "non-negativity of the inverse is not available")
    u = np.linalg.solve(L, np.ones(len(L)))
    if (u <= 0).any():
        raise ValueError(
            "no positive witness u with L u >= 1: L is not a certified "
            "nonsingular M-matrix, so L^-1 >= 0 is unproven")
    for _ in range(8):
        w, e = _fl_matvec(L, u)
        lo = float((w - e).min())
        if lo >= 1.0:
            return u
        if lo <= 0.0:
            raise ValueError(
                "no positive witness u with L u >= 1: L is not a certified "
                "nonsingular M-matrix, so L^-1 >= 0 is unproven")
        u = u * ((1.0 + 8.0 * float(e.max())) / lo)
    raise ValueError("witness normalization did not settle: L u >= 1 could "
                     "not be verified after eight rescalings")


def keff_bracket(L: np.ndarray, F: np.ndarray, phi: np.ndarray,
                 u: np.ndarray = None) -> Certified:
    """A two-sided bracket on a reactor's criticality eigenvalue k_eff,
    from any strictly positive trial flux. Collatz-Wielandt on
    A = L^-1 F: form psi = A phi and take the smallest and largest of
    the ratios psi_i / phi_i. A bad trial flux makes the bracket wider
    and cannot make it wrong. That is the same bargain the variational
    brackets strike, bought with positivity instead of a minimum
    principle.

    Applying A needs a linear solve, and the solve is inexact. That
    error is absorbed rather than assumed away, by the positivity
    already proven. With residual r = F phi - L psi_hat, the true psi
    differs from the computed one by L^-1 r. Since L^-1 >= 0, that
    difference is at most max|r| times the witness u, componentwise. So
    the same fact that licenses the theorem also prices the solver's
    mistake, and the bracket stays rigorous however sloppily psi was
    computed.

    Rounding in both matrix products is carried as well, and so is the
    arithmetic that forms the ratios, so what comes back is a bracket
    on the eigenvalue of the matrices exactly as stored. That those
    matrices discretize a reactor is a separate claim, and it belongs
    to whoever built them."""
    if (phi <= 0).any():
        i = int(np.argmin(phi))
        raise ValueError(
            f"trial flux is not strictly positive (phi[{i}] = "
            f"{phi[i]:g}): Collatz-Wielandt brackets nothing without a "
            "vector strictly inside the cone")
    if (F < 0).any():
        raise ValueError("fission operator has a negative entry: F >= 0 "
                         "is what makes A = L^-1 F non-negative")
    if u is None:
        u = mmatrix_witness(L)
    eps = np.finfo(float).eps
    b, be = _fl_matvec(F, phi)
    psi = np.linalg.solve(L, b)
    q, qe = _fl_matvec(L, psi)
    rho = _up(float((np.abs(b - q) + be + qe).max()) * (1.0 + 8.0 * eps))
    pad = rho * u
    ratio_lo, ratio_hi = (psi - pad) / phi, (psi + pad) / phi
    # the ratios are themselves three rounded operations deep (the pad's
    # multiply, the shift, the divide), so widen by more than that before
    # taking the extremes
    slop = 4.0 * eps * float(np.abs(ratio_hi).max())
    lo = _dn(float(ratio_lo.min()) - slop)
    hi = _up(float(ratio_hi.max()) + slop)
    # the midpoint rounds too, and near the floor the half-width is far
    # smaller than an ulp of the midpoint, so err has to carry that or
    # value +- err would not actually be [lo, hi]
    mid = 0.5 * (hi + lo)
    return Certified(mid, _up(_up(0.5 * (hi - lo)) + 2.0 * eps * abs(mid)),
                     Tier.RIGOROUS,
                     (f"keff-cw n={len(L)} collatz-wielandt "
                      f"m-matrix-witness solve-residual={rho:.3g} +fp",))


def slab_reactor(N: int = 100, width: float = 70.0, D1: float = 1.4,
                 D2: float = 0.4, sr1: float = 0.030, sa2: float = 0.100,
                 ss12: float = 0.020, nf1: float = 0.007,
                 nf2: float = 0.130) -> dict:
    """A two-group diffusion reactor in one dimension. This is the
    smallest model that is still a reactor rather than a generic
    eigenvalue problem. Fast neutrons are born from fission, then leak
    and slow down. Thermal neutrons arrive only by slowing down, and
    they are where most of the fissioning happens. That one-way traffic,
    down-scatter with no path back up, is what makes the loss operator
    non-symmetric.

    Cell-centred finite volume, zero flux at both edges, cross sections
    in the usual light-water range. Cheap enough to build dense, and
    small enough that the truth can be had independently from a dense
    eigensolver when the tests want to argue with the certificate."""
    h = width / N
    I, Z = np.eye(N), np.zeros((N, N))

    def diffusion(D):
        M = np.zeros((N, N))
        for i in range(N):
            M[i, i] = 2 * D / h ** 2
            if i > 0:
                M[i, i - 1] = -D / h ** 2
            else:
                M[i, i] += D / h ** 2      # ghost cell: phi_0 = -phi_1
            if i < N - 1:
                M[i, i + 1] = -D / h ** 2
            else:
                M[i, i] += D / h ** 2
        return M

    return {"L": np.block([[diffusion(D1) + sr1 * I, Z],
                           [-ss12 * I, diffusion(D2) + sa2 * I]]),
            "F": np.block([[nf1 * I, nf2 * I], [Z, Z]]),
            "N": N, "width": width,
            "label": f"slab {width:g}cm N={N}"}


def sn_slab_reactor(N: int = 60, width: float = 10.0, nang: int = 8,
                    st: float = 1.0, ss: float = 0.6,
                    nsf: float = 0.45) -> dict:
    """The same query without the diffusion approximation: one-group
    discrete-ordinates transport in a slab, which is the equation
    diffusion is an approximation TO. Neutrons are tracked by direction
    as well as position, on a Gauss-Legendre angular quadrature, and
    the width is in mean free paths.

    Nothing about the certificate changes, and that is the point.
    Streaming is differenced upwind, using step differencing. That
    scheme exists precisely because it cannot produce a negative flux,
    which is exactly the statement that the operator stays a Z-matrix.
    So mmatrix_witness and keff_bracket apply here unaltered, with no
    new proof and no new code. A certificate hung on a cone rather than
    on a quadratic form does not care which equation it is looking at,
    only whether the equation respects the cone. Diamond differencing
    would break this, and it should, because it is the scheme that can
    return negative fluxes.

    Cost is cells times angles, so this is dense-solvable in a slab and
    would not be in a reactor. It is kept small on purpose. It is here
    to show the archetype transferring, not to compete with a production
    transport code."""
    mu, w = np.polynomial.legendre.leggauss(nang)
    w = w / w.sum()             # so sum_b w_b psi_b is the isotropic average
    h = width / N
    n = N * nang
    L, F = np.zeros((n, n)), np.zeros((n, n))
    for a in range(nang):
        for c in range(N):
            i = c * nang + a
            L[i, i] += st + abs(mu[a]) / h          # collision + outflow
            up = c - 1 if mu[a] > 0 else c + 1      # the upwind neighbour
            if 0 <= up < N:                         # vacuum at both edges
                L[i, up * nang + a] -= abs(mu[a]) / h
            for b in range(nang):                   # isotropic in the lab
                L[i, c * nang + b] -= ss * w[b]
                F[i, c * nang + b] += nsf * w[b]
    return {"L": L, "F": F, "N": N, "width": width, "nang": nang,
            "label": f"S{nang} slab {width:g}mfp N={N}"}


def slab_buckling_keff(width: float, D1: float = 1.4, D2: float = 0.4,
                       sr1: float = 0.030, sa2: float = 0.100,
                       ss12: float = 0.020, nf1: float = 0.007,
                       nf2: float = 0.130) -> float:
    """The continuum answer for the same slab, in closed form. The
    fundamental mode of a bare slab is a half cosine, so the Laplacian
    is just multiplication by the buckling B^2 = (pi/width)^2 and the
    two-group balance collapses to arithmetic. Not a certificate and
    not used by one -- it is the independent truth the discretized
    model is measured against, so that the gap between the model and
    the world stays stated rather than implied."""
    B2 = (math.pi / width) ** 2
    return (nf1 + nf2 * ss12 / (sa2 + D2 * B2)) / (sr1 + D1 * B2)


# ------------------------------------------------- semiconductor
# devices. Every certificate above answers "how far is this number
# from the right one". This section answers a question nobody upstream
# has had to ask: is there a right one at all?
#
# A device simulator solves a nonlinear system by Newton's method and
# stops when the residual looks small. That is not a bound, and the
# gap is not pedantic. A small residual is consistent with a solution
# nearby, with a solution far away, and with no solution at all --
# residuals are small near near-singular points too, which is exactly
# where semiconductor equations live, since the carrier densities are
# exponentials of the unknown and swing twenty orders of magnitude
# across a junction. What is wanted is a theorem that converts a
# computed residual into the existence of an exact solution and a
# radius around the iterate that provably contains it. Kantorovich's
# theorem is that converter, and it needs three numbers: a bound on
# the inverse Jacobian, the size of the Newton step, and a Lipschitz
# constant for the Jacobian nearby.
#
# The first of those is the reactor's problem again. Discretize the
# nonlinear Poisson equation and the Jacobian comes out a Z-matrix --
# a positive second difference off the diagonal, a positive carrier
# response on it -- so mmatrix_witness, written to prove a reactor's
# operator inverse-positive, prices ||J^-1|| here without a line of
# new proof. The physics is unrelated. The cone is the same.

_VT = 0.025852          # thermal voltage kT/q at 300 K, volts
_NI = 1.0e10            # intrinsic carrier density of silicon, cm^-3
_EPS_SI = 11.7 * 8.854e-14   # permittivity of silicon, F/cm
_QE = 1.602176634e-19   # elementary charge, C


def inverse_bound(J: np.ndarray):
    """A rigorous upper bound on ||J^-1||_inf, by whichever of two
    routes the matrix admits. Returns the bound and the name of the
    route that produced it, because which one answered is part of the
    provenance.

    The first route is the reactor's witness. When J is a nonsingular
    M-matrix the bound is not merely valid, it is essentially exact.
    J^-1 >= 0 makes the row sums of |J^-1| equal to J^-1 e, so
    ||J^-1||_inf is literally ||J^-1 e||_inf, and that is what the
    witness computes. Measured on twelve junction Jacobians it lands
    on the true norm to every digit printed.

    The second route asks nothing about signs. Take any approximate
    inverse R, and if ||I - RJ||_inf = alpha < 1 then the Neumann
    series for (RJ)^-1 converges, J is nonsingular, and
    ||J^-1|| <= ||R|| / (1 - alpha). The cost is a matrix inverse and
    two matrix products, so it is a factor of n dearer than the
    witness. It is also the only route open once the Jacobian has a
    positive off-diagonal, which happens with a fourth-order stencil,
    or with the consistent mass matrix that a Grad-Shafranov pressure
    profile subtracts from its stiffness.

    Both the product and the row sums are computed in floating point,
    so both are padded. |fl(RJ) - RJ| <= gamma_n |R||J| entrywise with
    gamma_n = n u / (1 - n u), and 4 n eps dominates that with a
    factor of eight to spare, enough to also swallow the rounding of
    the pad's own matrix product. The row sums then round by at most
    gamma_n again, which is what the final slack covers. Rounding the
    denominator down keeps the quotient an upper bound."""
    try:
        return _up(float(mmatrix_witness(J).max())), "m-matrix-witness"
    except ValueError:
        pass
    n = len(J)
    eps = np.finfo(float).eps
    try:
        R = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        # LAPACK found an exact zero pivot. That is a refusal like any
        # other and gets the library's phrasing rather than numpy's,
        # since a caller distinguishing "no bound" from "crashed"
        # should not have to know which library was asked.
        raise ValueError(
            "neither route to ||J^-1||: J is not a certified M-matrix, "
            "and no approximate inverse exists to start a Neumann "
            "series -- the factorization hit an exact zero pivot, so J "
            "is singular as stored") from None
    P = R @ J
    pad = 4.0 * n * eps * (np.abs(R) @ np.abs(J))
    slack = 1.0 + 2.0 * n * eps
    alpha = _up(float((np.abs(np.eye(n) - P) + pad).sum(1).max()) * slack)
    if alpha >= 1.0:
        raise ValueError(
            f"neither route to ||J^-1||: J is not a certified M-matrix, "
            f"and the approximate inverse gives ||I - RJ|| = {alpha:.4g} "
            ">= 1, so the Neumann series is not proven to converge and J "
            "is not proven nonsingular")
    rnorm = _up(float(np.abs(R).sum(1).max()) * slack)
    return _up(rnorm / _dn(1.0 - alpha)), "rump-neumann"


def newton_enclosure(res: float, J: np.ndarray, lip):
    """Kantorovich's theorem as a checkable predicate: turn a computed
    residual into a proof that an exact solution EXISTS, and a radius
    around the iterate that contains it. Returns the radius and the
    route that priced the inverse Jacobian.

    Given a bound res on ||F(x)||_inf, a bound beta on ||J(x)^-1||_inf,
    and a constant K with ||J(a) - J(b)|| <= K |a - b| throughout a
    ball, set eta = beta * res and h = beta * K * eta. If h <= 1/2 then
    Newton's method started at x converges, its limit is an exact root,
    and that root lies within (1 - sqrt(1 - 2h)) * eta / h of x. The
    conclusion is existence, not merely accuracy, which is why it can
    be trusted at a point no one has proven is near anything.

    beta comes from inverse_bound, which tries the cheap cone route
    first and falls back to the Neumann one. Nothing here cares which
    answered. The theorem never asked for a Z-matrix. That was a
    restriction inherited from the only pricing method on hand.

    K is supplied by the caller as a function of the ball radius,
    because no generic code can know it. It is a second-derivative
    bound on the specific equations. The ball is then searched from
    tight to loose, and the first radius whose K certifies an enclosure
    no larger than the ball itself is the answer. Refuses when none
    does, which is the honest report that the iterate is not yet
    provably near a solution however small its residual looks."""
    beta, route = inverse_bound(J)
    eta = _up(beta * _up(res))
    worst = None
    for ball in (1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 3.0):
        k = _up(float(lip(ball)))
        h = _up(_up(beta * k) * eta)
        worst = (ball, k, h)
        if h <= 0.5:
            # the theorem's radius is eta * (1 - sqrt(1-2h))/h, which
            # loses most of its digits to cancellation when h is small
            # -- exactly the regime a converged Newton step lands in.
            # Multiplying through by the conjugate gives the same
            # number as eta * 2/(1 + sqrt(1-2h)), with no subtraction
            # of nearly equal quantities, and shows in passing that the
            # factor is always between 1 and 2. Rounding the
            # denominator down keeps the radius an upper bound.
            r = _up(eta * (2.0 / _dn(1.0 + math.sqrt(1.0 - 2.0 * h))))
            if r <= ball:
                return r, route
    ball, k, h = worst
    raise ValueError(
        f"Kantorovich fails: beta={beta:.3g} eta={eta:.3g} K={k:.3g} "
        f"gives h={h:.3g} > 1/2 even on a ball of radius {ball:g}, so "
        f"no solution is proven to exist near this iterate; the residual "
        f"would have to fall below {0.5 / (beta * beta * k):.3g}")


def pn_junction(N: int = 200, length_um: float = 1.0, Na: float = 1e17,
                Nd: float = 1e17) -> dict:
    """A silicon pn junction, one dimension, abrupt, at room
    temperature. The unknown is the electrostatic potential; the
    carriers follow it through Boltzmann statistics, so Poisson's
    equation closes on itself and becomes nonlinear:

        -lambda^2 psi'' + (exp(psi - V) - exp(-psi) - C(x)) = 0

    in units where potential is measured in thermal volts kT/q,
    densities in the intrinsic concentration, and length in the device
    length. C(x) is the doping, negative in the p region and positive
    in the n region. V is the reverse bias, entering as the split
    between the two carrier quasi-Fermi levels. That is the standard
    depletion-regime model, exact only where the current is negligible,
    which is what reverse bias means.

    It is returned scaled so the residual is O(1). The equation is
    divided by lambda^2/h^2, leaving a plain second difference against
    a small multiple of the exponentials. That is not cosmetic.
    Undivided, the two terms differ by ten orders of magnitude, and the
    residual cannot be evaluated below 10^-5 in double precision.
    Kantorovich needs 10^-8 to close, so the certificate would fail for
    no reason but arithmetic."""
    L = length_um * 1e-4
    x = np.linspace(0.0, 1.0, N + 1)
    C = np.where(x < 0.5, -Na, Nd) / _NI
    lam2 = _EPS_SI * _VT / (_QE * _NI * L * L)
    h = 1.0 / N
    return {"N": N, "x": x, "h": h, "C": C, "lam2": lam2, "L": L,
            "a": h * h / lam2, "Na": Na, "Nd": Nd,
            "label": f"pn junction {length_um:g}um Na={Na:.0e} "
                     f"Nd={Nd:.0e} N={N}"}


def _poisson_residual(dev: dict, psi: np.ndarray, v: float):
    """Residual and a bound on its own rounding, signed so that the
    Jacobian is a Z-matrix. Assumes exp is faithful to one ulp, the
    same assumption the Interval class states."""
    N, a, C = dev["N"], dev["a"], dev["C"]
    en, ep = np.exp(psi - v), np.exp(-psi)
    lo, hi = math.asinh(C[0] / 2), math.asinh(C[N] / 2) + v
    r = np.empty_like(psi)
    r[1:N] = (-(psi[:-2] - 2 * psi[1:-1] + psi[2:])
              + a * (en[1:-1] - ep[1:-1] - C[1:-1]))
    r[0] = psi[0] - lo
    r[N] = psi[N] - hi
    mag = np.zeros_like(psi)
    mag[1:N] = (np.abs(psi[:-2]) + 2 * np.abs(psi[1:-1]) + np.abs(psi[2:])
                + a * (en[1:-1] + ep[1:-1] + np.abs(C[1:-1])))
    mag[0], mag[N] = abs(psi[0]) + abs(lo), abs(psi[N]) + abs(hi)
    return r, 8.0 * np.finfo(float).eps * mag


def _poisson_jacobian(dev: dict, psi: np.ndarray, v: float) -> np.ndarray:
    N, a = dev["N"], dev["a"]
    J = np.zeros((N + 1, N + 1))
    i = np.arange(1, N)
    J[i, i - 1] = -1.0
    J[i, i + 1] = -1.0
    J[i, i] = 2.0 + a * (np.exp(psi - v) + np.exp(-psi))[1:N]
    J[0, 0] = J[N, N] = 1.0
    return J


def junction_potential(dev: dict, volts: float, m: int,
                       cap: float = 3.0) -> np.ndarray:
    """m damped Newton steps from the charge-neutral guess. The damping
    is the one device simulators use: no node's potential may move more
    than cap thermal volts in a step, because the residual contains
    exp(psi) and an undamped first step overflows it. Nothing here is
    certified -- this is the proposer. What it returns is checked."""
    v = volts / _VT
    psi = np.arcsinh(dev["C"] / 2) + np.where(dev["x"] < 0.5, 0.0, v)
    for _ in range(m):
        r, _ = _poisson_residual(dev, psi, v)
        d = np.linalg.solve(_poisson_jacobian(dev, psi, v), -r)
        big = float(np.abs(d).max())
        psi = psi + d * min(1.0, cap / big) if big > 0 else psi
    return psi


def junction_charge_bracket(dev: dict, volts: float,
                            psi: np.ndarray) -> Certified:
    """The depletion charge per unit area, in nC/cm^2, with a bound
    that covers the distance to the exact discrete solution.

    Two steps, and the first is the one that matters. Kantorovich
    certifies a radius r around psi containing an exact solution of the
    discrete equations. The charge is a smooth functional of the
    potential, so its own error is r times the functional's gradient,
    bounded over the ball rather than evaluated at the iterate.

    What this does NOT cover is the mesh, and that is stated rather
    than implied. The bound is about the exact solution of the
    discretized equations, not of the differential equation. Measured on
    the default junction at 1 V, the mesh costs about 7 parts in 10^4 at
    N=100 and falls as h^2. A converged rung's certified radius is worth
    10^-8 nC/cm^2. That is six orders of magnitude apart, with the
    discretization the larger. Which of the two binds does depend on the
    rung. At the FIRST rung that certifies at all, the radius is worth
    0.78 nC/cm^2 and the mesh is the smaller of the two. Both are
    printed."""
    v = volts / _VT
    N, a, C, h = dev["N"], dev["a"], dev["C"], dev["h"]

    def lip(ball):
        # J differs from J(psi) only on the diagonal, and each entry is
        # a * (exp(psi-v) + exp(-psi)); its derivative bounds the
        # Lipschitz constant over the ball. The 10^-12 is slack for the
        # exponentials' own rounding, thousands of times what a
        # faithful exp can cost.
        return a * float((np.exp(psi - v + ball)
                          + np.exp(-psi + ball)).max()) * (1.0 + 1e-12)

    r, e = _poisson_residual(dev, psi, v)
    res = _up(float((np.abs(r) + e).max()))
    rad, route = newton_enclosure(res, _poisson_jacobian(dev, psi, v), lip)
    w = np.full(N + 1, h)
    w[0] = w[N] = h / 2
    keep = dev["x"] >= 0.5
    en, ep = np.exp(psi - v), np.exp(-psi)
    rho = C - en + ep
    q = float((rho[keep] * w[keep]).sum())
    # the gradient is summed over a few hundred terms and each is an
    # exp, so widen it by far more than the n*eps that costs
    grad = float((w[keep] * (np.exp(psi[keep] - v + rad)
                             + np.exp(-psi[keep] + rad))).sum()) * (1 + 1e-12)
    # the charge sum cancels hard: in the neutral region the doping and
    # the majority carriers are both about 10^7 and their difference is
    # nearly nothing, so the rounding is set by the TERMS and not by
    # the total. Bound it on the terms.
    terms = float((w[keep] * (np.abs(C[keep]) + en[keep] + ep[keep])).sum())
    scale = _QE * _NI * dev["L"] * 1e9      # scaled charge -> nC/cm^2
    err = _up(_up(rad * grad) * scale
              + 8.0 * np.finfo(float).eps * terms * scale)
    return Certified(q * scale, err, Tier.RIGOROUS,
                     (f"junction-charge n={N + 1} newton-kantorovich "
                      f"{route} radius={rad:.3g} kT/q +fp",))


def depletion_width_analytic(dev: dict, volts: float) -> float:
    """The textbook answer for the same junction, in nm. Assume the
    depletion region is swept perfectly clean of carriers and the
    neutral regions perfectly neutral; then the charge is the doping,
    Poisson integrates twice by hand, and the width is a square root.
    This is not a certificate and is not used by one. It is the
    independent truth the computed model is measured against. It should
    be close, and it should not be exact. The real transition is smooth
    over a few Debye lengths, so the true depletion charge is a little
    smaller than this. The gap must then shrink as reverse bias widens
    the region it is wrong about."""
    vbi = _VT * math.log(dev["Na"] * dev["Nd"] / (_NI * _NI))
    return math.sqrt(2 * _EPS_SI * (vbi + volts) / _QE
                     * (1.0 / dev["Na"] + 1.0 / dev["Nd"])) * 1e7


# ----------------------------------------- the existence certificate,
# aimed back at the tokamak. The Grad-Shafranov certificate far above
# is a Prager-Synge energy bound, and where it applies it says the
# stronger thing: it measures the distance to the CONTINUUM solution,
# which no algebraic certificate can see. It pays for that with two
# restrictions. The source has to be linear in psi, because the
# implicit coupling is closed by a Picard contraction; and the
# contraction factor has to stay below one, which caps how hard the
# plasma current may respond to its own flux.
#
# A real pressure profile breaks both, independently. Tokamak pressure
# peaks on the magnetic axis and falls away outward, so p'(psi)
# DECREASES with psi and the source is nonlinear. Take the exponential
# profile with decay scale psi0 in flux units.
#
# The contraction factor is max|S'| Rmax / (Rmin lam1), and max|S'| is
# c/psi0 rather than c, so a peaked profile is a steep one and the
# wall arrives psi0 times sooner: measured, the Picard certificate
# refuses above c = 0.47 where the linear source reached 2.34.
#
# The Jacobian stops being a Z-matrix, and not marginally. The source
# contributes minus S' times the consistent mass matrix, whose
# off-diagonal entries are integrals of products of non-negative basis
# functions and are therefore POSITIVE. A decreasing profile has
# S' < 0, so those positive entries are added to the stiffness
# matrix's negative ones, and 392 of them come out positive at n=16 --
# for every c > 0, including couplings the contraction would have been
# happy to accept. The cone route to ||J^-1|| is closed here by the
# shape of the profile, not by the strength of the coupling, which is
# why lifting the Z-matrix restriction had to come first.
#
# Kantorovich with the Neumann route to beta asks neither question,
# and certifies out to c = 16, a contraction factor of 32. The price
# is the junction's price: the enclosure holds an exact solution of
# the DISCRETE equations, and the distance to the continuum one is the
# mesh's, stated beside it rather than folded in.


def gs_nonlinear_certified(n: int = 16, c: float = 4.0, psi0: float = 0.2,
                           m: int = 4, degree: int = 1) -> dict:
    """A tokamak equilibrium whose pressure profile is a real one:
    peaked on the magnetic axis, decaying outward, so the source is a
    nonlinear function of the flux and no contraction certificate
    reaches it. FEniCSx proposes a Newton iterate, untrusted as usual.
    Kantorovich then proves an exact solution of the discrete equations
    exists nearby, and says how near.

    The profile is exponential in the flux, and it is manufactured
    about the Solov'ev polynomial so that psi_ex remains an exact
    solution of the NONLINEAR problem at every c. That costs the
    certificate nothing, since it never looks at psi_ex except as
    boundary data, and it buys an independent truth to measure the
    discretization against. That is the job slab_buckling_keff does for
    the reactor and depletion_width_analytic for the junction.

    Newton starts cold, with the interior at zero, because a warm
    start from psi_ex would begin at the answer and hide the ladder.
    From cold the ladder is real, and its length depends on the
    physics. Measured, two steps certify at c = 1 and at c = 4, and
    three at c = 16.

    The Dirichlet nodes are eliminated rather than constrained. Their
    values are data and not unknowns, so the system certified is the
    one on the interior nodes and the radius is a max-norm on those.
    What the radius does not cover is the mesh. That is measured second
    order in h, and it is worth 1.4e-3 in flux at n=16 against a
    converged rung's 1e-15. Twelve orders apart, with the
    discretization the larger, and both printed.

    Assembly quadrature is not carried, the same declaration the
    Prager-Synge path makes. Everything from the assembled residual
    and Jacobian onward is, including the rounding inside beta."""
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem
    import ufl
    if psi0 <= 0.0:
        raise ValueError("psi0 must be positive: it is the decay scale of "
                         "the pressure profile, in flux units")
    R0, W, H = 3.0, 1.0, 1.0
    a_c, b_c, d_c = 1.0 / 100, 1.0 / 108, 1.0 / 10
    Rmin, Rmax = R0 - W, R0 + W
    lam1 = math.pi ** 2 * (1.0 / (2 * W) ** 2 + 1.0 / (2 * H) ** 2)

    msh = dmesh.create_rectangle(MPI.COMM_WORLD,
                                 [[R0 - W, -H], [R0 + W, H]], [n, n],
                                 dmesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    R, Z = x[0], x[1]
    psi_ex = a_c * (R ** 2 - R0 ** 2) ** 2 + b_c * R ** 2 * Z ** 2 \
        + d_c * Z ** 2
    ex = fem.Function(V)
    ex.interpolate(fem.Expression(psi_ex, V.element.interpolation_points))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = fem.locate_dofs_topological(
        V, tdim - 1, dmesh.exterior_facet_indices(msh.topology))
    free = np.ones(len(ex.x.array), bool)
    free[bdofs] = False

    uh = fem.Function(V)
    uh.x.array[:] = ex.x.array          # boundary is data
    uh.x.array[free] = 0.0              # interior starts cold
    v = ufl.TestFunction(V)
    g0 = -((8 * a_c + 2 * b_c) * R ** 2 + 2 * d_c)   # -Delta* psi_ex
    src = g0 + c * (ufl.exp(-uh / psi0) - ufl.exp(-psi_ex / psi0))
    Fform = fem.form((ufl.inner(1.0 / R * ufl.grad(uh), ufl.grad(v))
                      - src / R * v) * ufl.dx)
    Jform = fem.form(ufl.derivative(
        (ufl.inner(1.0 / R * ufl.grad(uh), ufl.grad(v))
         - src / R * v) * ufl.dx, uh))
    mass = fem.assemble_vector(fem.form(v * ufl.dx)).array.copy()

    def residual():
        return fem.assemble_vector(Fform).array[free].copy()

    def jacobian():
        A = fem.assemble_matrix(Jform)
        A.scatter_reverse()
        return A.to_dense()[np.ix_(free, free)]

    for _ in range(m):
        uh.x.array[free] -= np.linalg.solve(jacobian(), residual())

    J = jacobian()
    res = _up(float(np.abs(residual()).max()))
    lo = float(uh.x.array.min())
    mfree = float(mass[free].max())

    def lip(ball):
        # J(a) - J(b) is minus the mass matrix weighted by
        # (S'(a) - S'(b))/R, and |S'(a) - S'(b)| <= max|S''| |a - b|
        # pointwise, because a P1 nodal basis is a partition of unity
        # of non-negative functions and the max-norm on coefficients
        # therefore bounds the max-norm on the function. Summing one
        # row of the mass matrix over every column gives the integral
        # of that basis function, so the whole bound is one assembled
        # vector. The 1e-12 is slack for the exponential's own
        # rounding, far more than a faithful exp can cost.
        s2 = abs(c) / (psi0 * psi0) * math.exp((ball - lo) / psi0)
        return s2 / Rmin * mfree * (1.0 + 1e-12)

    rad, route = newton_enclosure(res, J, lip)
    # |Q(psi*) - Q(psi_h)| = |sum_i (dc_i) integral phi_i| <= rad *
    # sum of the interior masses, since the basis functions are
    # non-negative and only interior coefficients can move
    Qh = float(mass @ uh.x.array)
    q_err = _up(rad * _up(float(mass[free].sum())))
    # the factor the Picard certificate would need for this profile;
    # its Lipschitz constant is max|S'|, not c
    theta = abs(c) / psi0 * math.exp(-lo / psi0) * Rmax / (Rmin * lam1)
    Q = Certified(Qh, q_err, Tier.RIGOROUS,
                  (f"gs-nonlinear n={n} c={c:g} psi0={psi0:g} m={m} "
                   f"newton-kantorovich {route} radius={rad:.3g} flux; "
                   f"discrete equilibrium, contraction theta={theta:.3g} "
                   f"would refuse; assembly quadrature not carried",))
    return {"Q": Q, "radius": rad, "route": route, "residual": res,
            "theta": theta, "psi_err": float(np.abs(uh.x.array
                                                    - ex.x.array).max()),
            "uh": uh, "msh": msh, "free": free, "J": J}


# ------------------------------------------------------------- The
# planner. Everything above this line is a library of certified
# rewrites; this section is the first piece of the compiler the vision
# promises. Given a tolerance, it searches the rewrites' declared
# ladders, runs the cheapest promising rung, and lets the certificate
# -- never the cost model -- decide what is true.


class Refusal(ValueError):
    """A refusal is a receipt, not an apology. It records every rung
    the planner ran, what each was predicted to cost, what it actually
    cost, what it measured, and the price of the cheapest thing it did
    not try. The measured cost sits beside the predicted cost so the
    cost models are auditable the same way the brackets made the
    physics auditable. It subclasses ValueError so every existing
    caller that catches ValueError keeps working."""

    def __init__(self, slug, tol, tried, next_price, context=""):
        self.slug, self.tol, self.context = slug, tol, context
        self.tried = tried    # (rewrite, knob, predicted, secs, verdict)
        self.next_price = next_price
        at = f" at {context}" if context else ""
        meas = ", ".join(
            (f"({k}, {v:.3g}, {s:.2g}s)" if isinstance(v, float)
             else f"({k}, {v}, {s:.2g}s)")
            for _, k, _, s, v in tried) or "none"
        super().__init__(
            f"{slug}: no rung within budget certifies tol={tol:g}{at}; "
            f"measured (knob, err, cost): {meas}; {next_price}")


@dataclass(frozen=True)
class Rewrite:
    """One way to answer a question, with a declared ladder of effort.
    knobs is the ladder, cheapest rung first. cost predicts what a
    rung will cost, in any units kept consistent within one front
    door. It is a guess and is never trusted. run executes a rung and
    returns a Certified, which is the only arbiter. A rung may itself
    raise, and that refusal is a measurement too, so the planner records
    it and moves on. price_beyond, when given, names what the rung past
    the ladder would cost, for the refusal receipt."""
    name: str
    knobs: Tuple[Any, ...]
    cost: Callable[[Any], float]
    run: Callable[[Any], Certified]
    price_beyond: Callable[[], str] = None


def _fit_jump(meas, tol, remaining):
    """Model-guided escalation. After two honest failures there is no
    reason to keep climbing one rung at a time. Fit the measured errors
    to geometric decay, err ~ A * rho^knob, which is a straight line in
    (knob, log err). Then jump to the first remaining rung the line
    predicts will land at tol/2. Aiming past the target absorbs a wobbly
    fit, and a wrong jump costs one extra run, because the certificate
    still arbitrates. Fall back to plain stepping whenever the data
    refuse the model: fewer than two points, errors not decreasing, or a
    slope that is flat or rising, or a ladder whose knobs are not
    numbers at all. That last one is not hypothetical: the 2-RDM ladder
    is D, DQ, DQG, and a fitted jump has nothing to fit there."""
    if len(meas) < 2:
        return remaining[0]
    es = [e for _, e in meas]
    if any(e2 >= e1 for e1, e2 in zip(es, es[1:])) or min(es) <= 0:
        return remaining[0]
    try:
        ks = [float(k) for k, _ in meas] + [float(k) for k in remaining]
    except (TypeError, ValueError):
        return remaining[0]
    ks = ks[:len(meas)]
    ys = [math.log(e) for e in es]
    kbar = sum(ks) / len(ks)
    ybar = sum(ys) / len(ys)
    den = sum((k - kbar) ** 2 for k in ks)
    if den == 0:
        return remaining[0]
    s = sum((k - kbar) * (y - ybar) for k, y in zip(ks, ys)) / den
    if not math.isfinite(s) or s >= 0:
        return remaining[0]
    need = (math.log(tol / 2.0) - (ybar - s * kbar)) / s
    for k in remaining:
        if float(k) >= need:
            return k
    return remaining[-1]


def plan(slug: str, tol: float, rewrites, jump: bool = True,
         context: str = "", _composed: bool = False) -> Certified:
    """The planner. Seed a frontier with the first rung of every
    rewrite, priced by each one's own cost model, and run whichever
    rung is predicted cheapest. If its certificate meets the
    tolerance, that rewrite wins, and the plan trace is appended to
    the certificate's provenance. If not, the measurement is kept and
    the rewrite's next rung goes back on the frontier, chosen by
    _fit_jump when jumping is on. Then the next-cheapest promise runs.
    That promise may belong to a competing rewrite, so a rewrite that
    measures badly is dethroned the moment its next rung gets
    expensive. Cost models decide only the order of attempts, and
    certificates decide what is true. When every ladder is exhausted,
    the planner refuses with the full receipt.

    Every rung's predicted and measured cost is logged as structure,
    in the receipt field of the winning certificate and the tried field
    of the Refusal. So the cost models are auditable, and every run is
    calibration data for better ones on the machine that produced it.
    Measured seconds are the honest unit and the only one here.
    Dividing them by a fixed microbenchmark to make them portable was
    tried and measured not to work. A short benchmark and a long rung
    do not respond alike to contention or to cache state (see the
    compiler case page). Portability is a job for a cost model over
    the rewrite's own parameters, not for a yardstick. The provenance
    trace itself stays deterministic, because provenance is part of the
    certificate while timings are data about one run of it.

    A rewrite whose knob is a tuple is not a ladder. It is an
    assignment across several stages, and building one by hand is the
    debt this planner spent a while carrying. Four front doors each
    wrote their own allocation, their own product ladder, and their
    own sentence about which stage was short. So the planner refuses
    one now unless compose built it. The rule is not a style
    preference. A hand-rolled assignment cannot derive its binding
    stage, cannot share a node between branches, and cannot return a
    solved stage's unspent share to the stages after it. All three are
    properties of the graph, and a front door that flattens the graph
    into a list of tuples has thrown the graph away."""
    if not _composed:
        for rw in rewrites:
            if rw.knobs and isinstance(rw.knobs[0], (tuple, list)):
                raise ValueError(
                    f"plan({slug!r}): rewrite {rw.name!r} has a "
                    f"multi-dimensional knob {rw.knobs[0]!r}, which is a "
                    "budget split across stages rather than one ladder. "
                    "Declare it with Stage(...) and call compose(), which "
                    "derives the split, the binding stage and the sharing "
                    "that a hand-rolled product ladder cannot")
    frontier = []
    state = []
    for i, rw in enumerate(rewrites):
        state.append(list(rw.knobs))
        if rw.knobs:
            heapq.heappush(frontier,
                           (float(rw.cost(rw.knobs[0])), i, rw.knobs[0]))
    tried, meas = [], [[] for _ in rewrites]
    while frontier:
        cost, i, knob = heapq.heappop(frontier)
        rw, remaining = rewrites[i], state[i]
        while remaining and remaining[0] != knob:
            remaining.pop(0)          # rungs the jump skipped are gone
        if remaining:
            remaining.pop(0)
        t0 = time.perf_counter()
        try:
            c = rw.run(knob)
            verdict = c.err
        except ValueError as exc:
            c, verdict = None, f"raised: {exc}"
        tried.append((rw.name, knob, cost, time.perf_counter() - t0, verdict))
        if c is not None:
            if c.err <= tol:
                rejected = ", ".join(
                    f"{n}@{k}" for n, k, *_ in tried[:-1]) or "none"
                trace = (f"plan {slug}: tol={tol:g}; chose "
                         f"{rw.name}@{knob} (predicted {cost:g}); tried "
                         f"{len(tried)} rung{'s' if len(tried) != 1 else ''}; "
                         f"rejected {rejected}")
                return replace(c, provenance=c.provenance + (trace,),
                               receipt=tuple(tried))
            meas[i].append((knob, c.err))
        if remaining:
            nxt = _fit_jump(meas[i], tol, remaining) if jump \
                else remaining[0]
            heapq.heappush(frontier, (float(rw.cost(nxt)), i, nxt))
    prices = [rw.price_beyond() for rw in rewrites if rw.price_beyond]
    raise Refusal(slug, tol, tried,
                  "; ".join(prices) or "no rung remains", context)


@dataclass(frozen=True)
class Stage:
    """One node of a composed plan: a rewrite with its own ladder of
    effort, named so the assembly can find its certificate. run(knob)
    returns that node's certificate and cost(knob) its declared price,
    in whatever unit the front door chose -- one unit per plan, since
    the stage prices are summed.

    solve is the difference between a stage that must be searched and
    one that can be reasoned about. Given a slice of the error budget
    and the knobs already chosen upstream, solve returns the cheapest
    knob whose contribution fits, or None if none does. A stage that
    can answer that does not belong in a product ladder: the h-chain
    window cannot (what a width certifies is whatever the run
    measures), while a Legendre tail priced by orthogonality or a
    sample count following the 1/sqrt(m) law can, and searching those
    would be absurd -- m runs to a million. share is the fraction of
    the remaining budget such a stage may claim, left below 1 where
    the estimate it solves against is a guess worth padding.

    predict is the other half, and it is independent of solve: what
    this stage will spend of the budget at a given knob. An enumerated
    stage can still be predictable -- a mesh is chosen from a ladder,
    not solved for, yet its solve bill is guessed first-order from a
    pilot -- and declaring that is what leaves the right slack for the
    stages after it. Stages are consulted in declaration order, which
    is the order the budget is spent.

    inputs names the stages this one consumes, and it is what makes
    the plan a graph rather than a list. run is called with the knob
    and with those inputs' certificates, so a stage along a line can
    work from its predecessor's value. It also decides what may be
    shared: a node is memoized on its own knob AND on its inputs'
    knobs, so the two branches of a fan-in, which name no inputs, are
    each computed once however many pairs get walked -- that memo is
    what makes the product ladder cost the sum of the branches rather
    than their product -- while a mesh that consumed a profile is
    recomputed when the profile changes, as it must be.

    run may be None, and a stage without one is a choice that costs
    budget without producing a certificate of its own. A resolution is
    the example: picking an N-point kernel spends part of the error
    budget on what that kernel cannot resolve, but there is no
    separate object to hand back -- the bill lands inside the
    certificate of the stage that consumes the choice. Such a stage
    still declares knobs, a cost, and a predicted bill, and naming it
    in another stage's inputs delivers its chosen KNOB where a
    certificate-producing stage would deliver its certificate. That is
    what lets two stages share one discretization: both name it, and
    the memo keys them on it, so changing it recomputes both.

    inside goes with run=None and names the stage whose certificate
    already carries this stage's bill. Naming a choice as an input is
    not the same as paying for it: in the three-stage spectral plan
    both the model and the measurement are computed on the kernel's
    grid, but only the measurement's certificate contains the
    smearing error. Without that distinction the credit for buying
    the kernel down is handed to every consumer, and the refusal
    names the wrong stage."""
    name: str
    knobs: Tuple
    run: Callable
    cost: Callable
    solve: Callable = None
    predict: Callable = None
    share: float = 1.0
    inputs: Tuple = ()
    inside: str = None


def _binding_stage(certs, assemble, bills=(), consumes=()):
    """Which node's error is doing the damage. Zero each stage's error
    in turn, reassemble, and see how far the composed error falls: the
    stage whose removal helps most is the one worth buying next. This
    needs no knowledge of the graph's shape. Along a line the drop is
    the stage's error converted through the sensitivity that carries
    it; at a fan-in it is simply that branch's own error; and the
    assembly does that arithmetic either way, so the same three lines
    answer for both. It is also free -- every certificate is already
    in hand and reassembly is arithmetic, not computation.

    A stage that produces no certificate cannot be zeroed this way,
    because its bill is already inside somebody else's error and
    removing it would need that stage recomputed. Its predicted bill
    is passed in as bills and stands in for the drop, which is the
    right comparison anyway: the question is what buying that stage
    down would be worth.

    Which is also why the consumer has to give that bill back. Zeroing
    the certificate a resolution's bill lives inside removes the bill
    along with everything else, and the stage that merely HOLDS the
    smearing error would be credited with it -- twice over, once here
    and once as the resolution's own bill, so the parts sum past the
    whole and the wrong stage is named. Subtracting the bills a stage
    consumes fixes both: measured on the smeared-spectral plan it
    turns a drop of 0.178 into 0.0849, which is the statistics bill
    exactly, and moves the binding stage from the sample count to the
    kernel, which is the one that was actually short."""
    total = assemble(certs).err
    drops = dict(bills)
    paid = dict(consumes)
    for name, c in certs.items():
        bare = assemble({**certs, name: replace(c, err=0.0)}).err
        drops[name] = total - bare - sum(drops.get(p, 0.0)
                                         for p in paid.get(name, ()))
    return max(drops, key=drops.get), drops


def compose(slug: str, tol: float, stages, assemble: Callable,
            context: str = "", cost: Callable = None,
            wall: Callable = None) -> Certified:
    """A plan over a graph of certificates rather than a single ladder.

    The four composed plans in this library are wired by hand, each
    inside its own front door, and between them they say what a
    general combinator has to be. It cannot be a pipeline, because one
    of them is a fan-in: the object is a graph, with sensitivities on
    the edges that carry error and plain addition where two branches
    meet. And it cannot assume the split is computable in advance,
    because for one of them it is not.

    So the graph is declared as stages and one assemble function, and
    composition stays where it already worked -- assemble receives the
    stages' certificates by name and combines them with `through`, `+`
    or `-`, whichever the graph calls for. What this adds is the
    search. Stages that can solve for a knob given a budget are
    solved, in declaration order, each taking its share of what is
    left; stages that cannot are enumerated, their product priced by
    the sum of the stage costs and walked cheapest-first by plan().
    Sharing the node results across the walk is what keeps that sum
    from becoming a product: escalating one branch reuses the other.

    The refusal names the binding stage, and derives it rather than
    being told: see _binding_stage. Every front door here used to
    write that sentence by hand, and hand-written it can only describe
    the shape its author had in mind.

    Two things a front door may still say for itself, because no
    generic rule knows them. cost overrides how stage prices combine:
    the default sums them, which is right when the stages are separate
    pieces of work, and wrong when they multiply -- measuring an
    N-point correlator m times costs N*m, not N+m, and the walk order
    that follows from adding them is a different walk. wall overrides
    the sentence printed when no rung exists at all. The derived one
    runs the graph at its widest declared rung to see where the error
    floors, which is affordable for a mesh ladder and absurd for a
    sample count whose ceiling is a million draws; a front door that
    knows its wall in closed form should say so and skip the run.

    A note on what this does not do, now with three measurements behind
    it rather than one. The search walks the cost ladder, and three
    attempts to make it use the graph's shape instead have all been
    built and all been abandoned.

    Escalate the binding stage rather than stepping: reached a
    certifying pair in 4-7 runs where the ladder takes 8-16, but in
    the thing that costs -- nodes computed, not assemblies, since the
    branches are shared -- it was 5 against 6, then 6 against 6, then
    7 against 6, and it twice returned a dearer assignment because it
    never revisited what it stepped over.

    Order by MARGINAL node cost, so that no node is ever bought while
    an untried assembly of already-computed nodes remains: saved one
    node at one tolerance out of eight, and at that tolerance returned
    a dearer assignment. Physics ladders are geometric in cost, so the
    newest node dominates the total and the two orders almost coincide.

    Prune an assignment whose already-measured components exceed the
    tolerance: admissible, it returned the same assignment every time,
    and it cuts the walk hard -- 1024 assemblies to 5 on a five-branch
    graph. It saves no nodes. Zero of them in nine of twelve cases
    across two to five branches, because the cost-ordered walk reaches
    the cheap knobs of every branch before any pruning can bite.

    All three fail for one reason, and it is the memo. Sharing nodes
    already makes node computations linear in the graph's size, so the
    only quantity left for a cleverer search to reduce is assemblies,
    and assemblies are arithmetic on certificates already in hand.
    That would change if a graph appeared whose cheap rungs did not
    already cover every knob -- if coarse were dear and fine were
    cheap, which is the opposite of every ladder in this library."""
    solved = {}
    at = {st.name: i for i, st in enumerate(stages)}
    order = _topo(stages)
    paper = tuple(st.name for st in stages if st.run is None)

    def certs_at(assign):
        """Evaluate the graph in edge order, which is not the order the
        stages were declared in: declaration order is the order the
        BUDGET is spent, and the two need not agree. A mesh whose bill
        is predicted first may still consume a profile computed
        before it. Each node is keyed on its own knob and its inputs',
        so independent branches are shared and dependent ones are
        recomputed when what they consumed changes.

        A stage with no run contributes its knob instead of a
        certificate, which is what a shared discretization is: two
        stages name it, both see the same number, and both are
        recomputed when it changes."""
        out = {}
        for st in order:
            k = assign[at[st.name]]
            if st.run is None:
                out[st.name] = k
                continue
            key = (st.name, k, tuple(assign[at[i]] for i in st.inputs))
            if key not in solved:
                solved[key] = st.run(k, {i: out[i] for i in st.inputs})
            out[st.name] = solved[key]
        return out

    def real(out):
        return {n: c for n, c in out.items() if n not in paper}

    searched = [st for st in stages if st.solve is None]

    def fill(chosen):
        """Spend the budget in declaration order. An enumerated stage
        reports what its knob will cost; a solvable one is handed its
        share of what remains and returns the cheapest knob that
        fits. Either can run the budget out, and that is a rung that
        never existed rather than a rung that fails.

        A solved stage is charged what it actually costs when it can
        say -- a knob is chosen as the cheapest that FITS its share,
        so it usually costs less, and returning the difference to the
        pool is free tightening for the stages after it. Without a
        predict there is nothing to charge but the whole share."""
        left = tol
        for st in stages:
            if st.name not in chosen:
                budget = st.share * left
                k = st.solve(budget, chosen)
                if k is None:
                    return None
                chosen = {**chosen, st.name: k}
                left -= st.predict(k, chosen) if st.predict else budget
            elif st.predict:
                left -= st.predict(chosen[st.name], chosen)
            if left <= 0:
                return None
        return chosen

    rungs = []
    for combo in _product(tuple(st.knobs for st in searched)):
        chosen = fill(dict(zip((st.name for st in searched), combo)))
        if chosen is not None:
            rungs.append(tuple(chosen[st.name] for st in stages))

    def price(assign):
        if cost:
            return float(cost(dict(zip((st.name for st in stages), assign))))
        return sum(st.cost(k) for st, k in zip(stages, assign))

    def bills_at(assign):
        chosen = dict(zip((st.name for st in stages), assign))
        return {st.name: st.predict(chosen[st.name], chosen)
                for st in stages if st.name in paper and st.predict}

    eats = {}
    for st in stages:
        if st.run is None and st.inside:
            eats.setdefault(st.inside, []).append(st.name)

    def run(assign):
        out = certs_at(assign)
        certs = real(out)
        done = assemble(certs)
        _, drops = _binding_stage(certs, assemble, bills_at(assign), eats)
        note = (f"{slug} split at "
                + " ".join(f"{st.name}={k}"
                           for st, k in zip(stages, assign))
                + ": " + " + ".join(f"{n} {d:.3g}" for n, d in drops.items())
                + f" against tol={tol:g}")
        return replace(done, provenance=done.provenance + (note,))

    def beyond():
        if wall:
            return wall()
        top = tuple(st.knobs[-1] for st in stages)
        certs = real(certs_at(top))
        who, drops = _binding_stage(certs, assemble, bills_at(top), eats)
        return ("every stage is at its widest declared rung ("
                + ", ".join(f"{st.name}={k}" for st, k in zip(stages, top))
                + f"), where the composed error floors at "
                f"{assemble(certs).err:.3g}; {who} is the binding stage, "
                f"contributing {drops[who]:.3g} of that")

    rw = Rewrite("+".join(st.name for st in stages),
                 tuple(sorted(rungs, key=price)), price, run, beyond)
    return plan(slug, tol, [rw], jump=False, context=context,
                _composed=True)


def _product(ladders):
    """itertools.product, spelled out to keep the module's imports as
    short as its dependency list."""
    out = [()]
    for ladder in ladders:
        out = [row + (k,) for row in out for k in ladder]
    return out


def _topo(stages):
    """Evaluation order for the graph: every stage after the stages it
    names as inputs. Declaration order cannot serve, because that is
    the order the error budget is spent and a stage may well be
    charged before the stage it consumes is computed."""
    done, order, todo = set(), [], list(stages)
    while todo:
        ready = [st for st in todo if all(i in done for i in st.inputs)]
        if not ready:
            raise ValueError("compose: stage inputs do not form a DAG "
                             f"({', '.join(st.name for st in todo)} "
                             "cannot be ordered)")
        for st in ready:
            order.append(st)
            done.add(st.name)
            todo.remove(st)
    return tuple(order)


def heisenberg_energy_dispatch(N: int, tol: float,
                               correction_iters: int = 10,
                               dense_max: int = 12, ell_max: int = None,
                               jump: bool = True) -> Certified:
    """Ground energy of the N-site spin-1/2 Heisenberg chain, certified
    so the bracket half-width per bond meets tol. This is the one front
    door where two rewrites compete, and competition is what forces
    them onto a single currency. Cost here is eigendecompositions times
    dimension. A window rung runs the multiplier ascent before it
    brackets anything, so it costs correction_iters x 2^ell. The dense
    bracket diagonalizes once at 2^N and is exact, so it enters the
    race only when the chain is small enough to form (N <= dense_max).
    Pricing the window at 2^ell alone dropped the iteration count and
    inverted the only comparison the library ever makes. It quoted the
    widest window at half the price of a dense rung that measures
    nearly twice as fast. The planner runs whichever promises cheapest,
    and the certificates decide. Returns the total-energy bracket, with
    the plan trace last in its provenance."""
    tol_total = tol * (N - 1)
    ells = tuple(range(2, min(N - 1, ell_max or 10) + 1))
    eighs = max(correction_iters, 1)      # the ascent, or one bracket

    def beyond():
        nxt = ells[-1] + 1
        return (f"the next window ell={nxt} costs {eighs} x 2^{nxt} = "
                f"{eighs * 2 ** nxt} and is past the declared ladder")

    rewrites = [Rewrite("window", ells, lambda ell: eighs * 2.0 ** ell,
                        lambda ell: heisenberg_chain_bracket(
                            N, ell, correction_iters), beyond)]
    if N <= dense_max:
        rewrites.append(
            Rewrite("dense", (N,), lambda _: 2.0 ** N,
                    lambda _: heisenberg_chain_bracket(N, ell=N)))
    return plan("chain-energy", tol_total, rewrites, jump=jump,
                context=f"N={N}")


def h_chain_energy_dispatch(n: int, tol: float, d: float = 1.8,
                            ell_max: int = 5,
                            jump: bool = True,
                            rdm2: bool = True) -> Certified:
    """Ground-energy bracket for the n-atom hydrogen chain, certified so
    the half-width per atom meets tol. Two rewrites race here.

    The window ladder answers with one knob, the window length ell, at
    cost 4^ell. That is quantum chemistry's folklore method ladder, run
    a cheap method and distrust it and run a dearer one, replaced by a
    declared ladder the planner climbs or jumps until a certificate says
    stop.

    The second rewrite is the 2-positivity relaxation of the 2-RDM,
    bounded above by a self-consistent determinant. Its knob is which
    conditions it imposes, from D through DQ to DQG, and it does not
    know what a window is. Measured at n = 4 to 8 its half width per
    atom is 8.0 to 8.6 mHa at DQG, which barely moves with the chain,
    while the window ladder at a fixed length loses ground as the chain
    grows. What it costs is 2.5 to 65 s against a window rung's
    fractions of a second.

    So the cheap window rungs run first and the relaxation gets its turn
    when they have all failed, which is the cost model doing its job.
    Certificates still decide.

    Both rewrites bound the same number. The window brackets the ground
    energy over every particle sector, and its upper half is an
    N-electron trial state, so its bracket contains the N-electron ground
    energy too. The relaxation brackets that sector directly. The query
    this front door answers is therefore the N-electron ground energy,
    and the window's answer is simply looser below it.

    h_chain_rdm2_bracket, which intersects the two, is still available
    and still reports which side won. It is not raced here because the
    window's block-product upper bound stopped winning once the
    determinant became self-consistent, measured at 147 mHa above exact
    against 62 at H4, so the pairing's ell knob had nothing left to buy.
    """
    tol_total = tol * n
    ells = tuple(range(2, min(n - 1, ell_max) + 1))
    win_memo, rdm_memo = {}, {}

    def window_at(ell):
        if ell not in win_memo:
            win_memo[ell] = h_chain_bracket(n, d, ell)
        return win_memo[ell]

    def beyond():
        nxt = ells[-1] + 1
        return (f"the next window ell={nxt} costs 4^{nxt} = {4 ** nxt} "
                f"and is past the declared ladder (ell_max={ell_max})")

    rewrites = [Rewrite("window", ells, lambda ell: 4.0 ** ell,
                        window_at, beyond)]

    if rdm2 and ells:
        # The semidefinite program is priced 13 n^4 in the window's own
        # currency of 4^ell. Exponent and constant are both measured, on
        # the chains n = 8 to 14 where the fixed overheads have washed
        # out, and like every cost model here it only orders the
        # attempts. Only the first rung is charged for it, because the
        # lower half does not depend on ell and is computed once. Both
        # rewrites also share the window memo, so a rung the other
        # already ran measures as nearly free, exactly the state
        # dependence the compiler page reports.
        T, V, eri, enuc = _h_chain_basis(n, d)
        rw = _rdm2_rewrite(T + V.sum(0), eri, n, enuc, n, 1e-7, 100_000)

        def guarded(cond, _run=rw.run):
            try:
                return _run(cond)
            except ImportError as exc:
                raise ValueError(
                    f"the 2-RDM rung needs cvxpy and scs: {exc}")

        rewrites.append(replace(rw, run=guarded))

    return plan("hchain-energy", tol_total, rewrites, jump=jump,
                context=f"n={n}")


def smeared_spectral_dispatch(measure: Callable, cov1: Callable,
                              omega: float, sigma: float, tol: float,
                              Ns=(8, 12, 16), m_max: int = 1 << 20,
                              z: float = 5.0) -> Certified:
    """The first composed plan: one error budget, two bills to pay.
    A smeared spectral value from noisy correlator data carries two
    error streams. The smearing bill c*C(1) is what the kernel
    reconstruction cannot resolve, and it shrinks only by buying more
    correlator times N. The statistics bill z*amp/sqrt(m) is what the
    noise obscures, and it shrinks only by buying more samples m. Here
    amp is the noise's reach into the answer, measured through the
    same g whose norm the certificate exports. One tolerance must cover
    both. For each rung N of the declared ladder the smearing bill is
    what it is, the leftover budget goes to statistics, and the sample
    count follows in closed form from the 1/sqrt(m) law. That is the
    marginal-cost balancing MLMC uses for level allocation, collapsed
    to a formula because one stage is continuous. Each rung is priced
    at N*m, which is an N-point correlator measured m times. The
    planner runs the cheapest promise and the certificate of the run
    decides, so a wrong pilot estimate costs extra rungs and cannot
    cost truth.

    measure(N, m) must return the m-sample mean correlator C(1..N), and
    cov1(N) the covariance of a single sample. The split aims at 80%
    of the leftover budget so a wobbly pilot C(1) does not push the
    first rung over.

    Wired through compose, and the shape it needed is the reason
    compose grew a stage that produces no certificate. There is only
    ONE certificate here, smeared_spectral's, and both bills are
    inside it. The kernel is not a second certificate to add on. It is
    a choice that spends budget and shows up in somebody else's error.
    Declaring it that way is what lets the sample count be solved
    against what the kernel leaves, which is the whole plan."""
    C1 = float(np.asarray(measure(Ns[0], 1), float)[0])

    def smearing(N, _chosen):
        return _hlt_solve(N, omega, sigma)[1] * C1

    def samples(budget, chosen):
        N = chosen["kernel"]
        g, c = _hlt_solve(N, omega, sigma)
        V = np.asarray(cov1(N), float)
        amp = c * math.sqrt(float(V[0, 0])) \
            + math.sqrt(float(g @ V[1:, 1:] @ g))
        m = max(1, math.ceil((z * amp / budget) ** 2))
        return m if m <= m_max else None

    def measured(m, up):
        N = up["kernel"]
        C = np.asarray(measure(N, m), float)
        cov = np.asarray(cov1(N), float) / m
        return smeared_spectral(C, omega, sigma, cov=cov, z=z)

    def wall():
        cN = _hlt_solve(max(Ns), omega, sigma)[1]
        return (f"the finest declared kernel (N={max(Ns)}) leaves a "
                f"smearing bill of about c*C(1)={cN * C1:.3g}; whatever "
                f"tol remains above it buys statistics at "
                f"(z*amp/slack)^2 samples, capped at m_max={m_max}")

    stages = (
        Stage("kernel", tuple(Ns), run=None, cost=float, predict=smearing,
              inside="samples"),
        Stage("samples", (m_max,), run=measured, cost=float,
              solve=samples, share=0.8, inputs=("kernel",)),
    )
    return compose("smeared-spectral", tol, stages,
                   lambda cs: cs["samples"],
                   context=f"omega={omega:g} sigma={sigma:g}",
                   cost=lambda ch: float(ch["kernel"]) * float(ch["samples"]),
                   wall=wall)


def gs_flux_dispatch(tol: float, c: float = 1.0, A: float = 0.4,
                     rho: float = 0.5, meshes=(8, 16, 32),
                     k_max: int = 12) -> Certified:
    """The first pipeline of two different rewrites under one budget.
    The query is the total poloidal flux of the coupled equilibrium
    whose current profile is the full declared Legendre series. No solve
    ever sees that source exactly. Two rewrites answer it together.
    legendre_source_profile truncates the series at k terms and
    certifies the dropped tail. gs_equilibrium_certified solves on a
    mesh of n cells and certifies its own discretization. The solve's
    exported sensitivity is the exchange rate between them, so total
    error = solve error + sensitivity * tail. One tolerance pays both,
    so per mesh rung the leftover budget prices the truncation. The
    tail curve is known in advance, by orthogonality and with no solve
    needed, so k follows by lookup. The mesh bill is only predicted,
    by first-order decay from a pilot, and must be re-certified by the
    run. Each rung costs n^3 + k, and the planner runs the cheapest
    promise. The composed certificate is the referee. It is chained by
    Certified.through, and its tier is the weakest claim in the
    chain."""
    pilot_prof = legendre_source_profile(A, rho, 2)
    r0 = gs_equilibrium_certified(n=meshes[0], c=c,
                                  source_coeffs=tuple(pilot_prof.value))
    S = r0["Q"].sensitivity.bound
    e0 = r0["Q"].err

    stages = (
        Stage("mesh", tuple(meshes),
              run=lambda n, up: gs_equilibrium_certified(
                  n=n, c=c, source_coeffs=tuple(up["profile"].value))["Q"],
              cost=lambda n: float(n) ** 3,
              predict=lambda n, _ch: e0 * meshes[0] / n,
              inputs=("profile",)),
        Stage("profile", tuple(range(1, k_max + 1)),
              run=lambda k, _up: legendre_source_profile(A, rho, k),
              cost=float,
              solve=lambda budget, _ch: next(
                  (k for k in range(1, k_max + 1)
                   if S * legendre_source_profile(A, rho, k).err <= budget),
                  None),
              share=0.8),
    )
    # profile is declared second because the budget is spent in that
    # order -- the mesh bill is predicted first and the truncation gets
    # a share of what survives -- but it is evaluated first, since the
    # mesh names it as an input and the graph runs in edge order
    return compose("gs-flux", tol, stages,
                   lambda cs: cs["mesh"].through(cs["profile"]),
                   context=f"c={c:g} A={A:g} rho={rho:g}")


def gs_continuum_flux(c: float = 4.0, psi0: float = 0.2,
                      meshes=(8, 16, 32, 64), m: int = 5) -> Certified:
    """The flux content of the equilibrium rather than of its mesh,
    for the nonlinear pressure profile that the Prager-Synge path
    cannot reach.

    This closes the honest weakness in gs_nonlinear_certified. That
    certificate is RIGOROUS and it is about the discrete equilibrium;
    Prager-Synge, where it applies, is RIGOROUS about the continuum
    one, which is the stronger claim. So the comparison was never
    like for like, and the missing statement is this one -- bought
    the only way it can be bought without a new theorem, by refining
    and reading the trend, which costs a tier.

    What comes back is worth putting beside the other two. The
    discrete radius at n=16 is 10^-15; the mesh is worth 10^-3; the
    continuum certificate is the second of those and EMPIRICAL, so
    the ordering is the opposite of the one the tier labels suggest.
    A weaker claim about the real thing beats a stronger claim about
    a mesh, whenever the mesh is what you were not asking about.

    Cost is the wall here rather than the physics. beta comes from a
    dense inverse of the interior block, so the finest rung is cubic
    in the number of nodes and n=64 is where that stops being
    pleasant."""
    rungs = [gs_nonlinear_certified(n=n, c=c, psi0=psi0, m=m)["Q"]
             for n in meshes]
    return continuum_limit(rungs, [1.0 / n for n in meshes],
                           "continuum poloidal flux: mesh ladder "
                           f"n={'/'.join(map(str, meshes))}")


def gs_exact_flux() -> float:
    """The continuum flux content in closed form, which exists only
    because the source was manufactured about the Solov'ev polynomial:
    psi_ex solves the nonlinear problem exactly at every coupling, so
    the exact answer is the integral of a polynomial over a rectangle
    and can be done by hand. Not a certificate and not used by one --
    it is the independent truth the mesh ladder is measured against,
    the job slab_buckling_keff does for the reactor.

    integral over [R0-W, R0+W] x [-H, H] of
    a (R^2 - R0^2)^2 + b R^2 Z^2 + d Z^2, with W = H = 1 and R0 = 3:
    the first term has no Z dependence and contributes 2a * 24.4, the
    second separates into (56/3)(2/3), the third into 2 * (2/3)."""
    a_c, b_c, d_c = 1.0 / 100, 1.0 / 108, 1.0 / 10
    return 2 * a_c * 24.4 + b_c * (56 / 3) * (2 / 3) + d_c * 2 * (2 / 3)


def pole_correlator(A: float, rho: float, E0: float, dE: float,
                    K: int, N: int) -> Certified:
    """A declared spectral model, truncated where you can afford to.
    The model is an infinite tower of poles: weight A*rho^k at energy
    E0 + k*dE, k = 0, 1, 2, ... Its Euclidean correlator is
    C(t) = sum_k A rho^k exp(-(E0 + k dE) t). Keep the first K poles,
    and the dropped tail at each time is itself a geometric series,
    because exp(-E_k t) with E_k linear in k IS geometric in k. So the
    truncation error is not estimated. It is summed in closed form.
    value is the truncated correlator on the grid t = 1..N. err is
    the 2-norm of the exact tail over that grid, which is the norm
    the smearing certificate's exported sensitivity converts."""
    if not (0.0 < rho < 1.0 and dE >= 0.0):
        raise ValueError("the pole tower must converge: 0 < rho < 1 "
                         "and dE >= 0")
    ts = np.arange(1, N + 1, dtype=float)
    ks = np.arange(K, dtype=float)[:, None]
    C = (A * rho ** ks * np.exp(-(E0 + dE * ks) * ts)).sum(axis=0)
    q = rho * np.exp(-dE * ts)
    tail = A * np.exp(-E0 * ts) * q ** K / (1.0 - q)
    # the formula is exact in real arithmetic; the 1e-12 pad covers
    # its float evaluation, which can round a few ulps below
    return Certified(C, _up(float(np.linalg.norm(tail)) * (1 + 1e-12)),
                     Tier.RIGOROUS,
                     (f"pole-model K={K} A={A:g} rho={rho:g} E0={E0:g} "
                      f"dE={dE:g} exact geometric tail",))


def spectral_pipeline_dispatch(sample: Callable, cov1: Callable,
                               A: float, rho: float, E0: float,
                               dE: float, omega: float, sigma: float,
                               tol: float, Ns=(8, 12, 16),
                               K_max: int = 64, m_max: int = 1 << 20,
                               z: float = 5.0) -> Certified:
    """A three-stage chain, with the last stage repricing the first.
    The query is the smeared spectral value of the full declared pole
    tower. There are infinitely many states, so no finite correlator
    ever holds them all. Three bills go against one tolerance. The model
    bill truncates the tower at K poles, with the tail priced exactly by
    pole_correlator. The statistics bill measures the truncated
    correlator m times through a noisy channel. The smearing bill is
    what the N-point kernel cannot resolve. The exchange rate between
    model error and answer error is the smearing certificate's
    exported sensitivity, and that constant depends on N, because a
    different kernel is a different linear functional with a
    different norm. So K is not a constant of the problem. Every rung
    of the resolution ladder reprices the model stage before it, and
    the same tail that fits one rung's budget can be over or under
    another's. That midstream repricing is what makes this a chain
    rather than three independent budgets.

    Per rung the split is closed-form. C(1) of the model is a
    geometric sum, so the smearing bill needs no pilot. A tenth of
    the leftover prices the model tail, which is exponentially cheap,
    so K grows only logarithmically. The rest buys samples by the
    1/sqrt(m) law, aimed at 80% to absorb wobble. sample(C, m) must
    return the m-sample noisy mean of the exact correlator C, and
    cov1(N) the single-sample covariance. Each rung costs N*m + K, and
    the planner runs the cheapest promise. The chained certificate is
    the referee, with the model tail converted by Certified.through.

    Wired through compose, and it is the plan that says why a stage
    may be a choice rather than a certificate. The kernel's grid is
    named by BOTH the model, whose tail is normed over t = 1..N, and
    the measurement, whose smearing bill it sets. That is one knob with
    two consumers, and the memo keys both on it, so changing the kernel
    recomputes both. The midstream repricing that makes this a chain
    rather than three budgets is then nothing special. The model stage
    solves against a budget that already has the kernel's bill taken
    out of it, so a different kernel hands it a different budget."""
    C1 = A * math.exp(-E0) / (1.0 - rho * math.exp(-dE))

    def rate(N):
        # mirrors smeared_spectral's exported sensitivity; drifting
        # from it mis-prices rungs and cannot cost truth, because the run
        # converts the tail through the certificate's own bound
        g, c = _hlt_solve(N, omega, sigma)
        return math.hypot(c, float(np.linalg.norm(g)))

    def smearing(N, _chosen):
        return _hlt_solve(N, omega, sigma)[1] * C1

    def tail(K, chosen):
        N = chosen["kernel"]
        return rate(N) * pole_correlator(A, rho, E0, dE, K, N).err

    def poles(budget, chosen):
        return next((K for K in range(1, K_max + 1)
                     if tail(K, chosen) <= budget), None)

    def samples(budget, chosen):
        N = chosen["kernel"]
        g, c = _hlt_solve(N, omega, sigma)
        V = np.asarray(cov1(N), float)
        amp = c * math.sqrt(float(V[0, 0])) \
            + math.sqrt(float(g @ V[1:, 1:] @ g))
        m = max(1, math.ceil((z * amp / budget) ** 2))
        return m if m <= m_max else None

    def measured(m, up):
        C = np.asarray(sample(up["model"].value, m), float)
        cov = np.asarray(cov1(up["kernel"]), float) / m
        return smeared_spectral(C, omega, sigma, cov=cov, z=z)

    def wall():
        cN = _hlt_solve(max(Ns), omega, sigma)[1]
        return (f"the finest declared kernel (N={max(Ns)}) leaves a "
                f"smearing bill of about c*C(1)={cN * C1:.3g}; the "
                f"model is not the wall (its tail falls geometrically, "
                f"K_max={K_max}); whatever tol remains buys statistics, "
                f"capped at m_max={m_max}")

    stages = (
        Stage("kernel", tuple(Ns), run=None, cost=float, predict=smearing,
              inside="samples"),
        Stage("model", tuple(range(1, K_max + 1)), cost=float,
              run=lambda K, up: pole_correlator(A, rho, E0, dE, K,
                                                up["kernel"]),
              solve=poles, predict=tail, share=0.1, inputs=("kernel",)),
        Stage("samples", (m_max,), run=measured, cost=float,
              solve=samples, share=0.8, inputs=("kernel", "model")),
    )
    return compose("spectral-pipeline", tol, stages,
                   lambda cs: cs["samples"].through(cs["model"]),
                   context=f"omega={omega:g} sigma={sigma:g}",
                   cost=lambda ch: (float(ch["kernel"]) * float(ch["samples"])
                                    + float(ch["model"])),
                   wall=wall)


def h_chain_gap_dispatch(n: int, tol: float, d_near: float = 1.8,
                         d_far: float = 3.0, ell_max: int = 5,
                         correction_iters: int = 60) -> Certified:
    """A budget split across a fan-in rather than along a chain.
    The other pipelines here are lines: one stage's answer becomes the
    next stage's input, and the sensitivity is the exchange rate that
    converts error along the line. This query has no line. It asks for
    the energy gap of the n-atom hydrogen chain between two geometries,
    meaning how much it costs to stretch the chain from d_near to
    d_far. That is a difference of two brackets which never see each
    other. Errors add, with no exchange rate to convert. The only
    question is how to divide one tolerance between two independent
    branches.

    That division cannot be done by formula here. Every other
    allocation in this library leans on something known before the run,
    such as a geometric tail summed exactly, a 1/sqrt(m) statistics
    law, or a first-order mesh decay fitted from a pilot. The h-chain
    window ladder offers none of them. What a window of width ell
    certifies is whatever the run measures. So the allocation is a
    search. Every pair of window widths is a rung, priced
    4^ell_near + 4^ell_far, and the planner walks the product ladder in
    cost order until a pair certifies. The certificates arbitrate,
    exactly as along a line.

    Two things follow that a line does not show. First, branches are
    shared, so a memo makes the product ladder cost the SUM of the two
    ladders rather than their product, because escalating one branch
    reuses the other. Second, the winning pair is asymmetric, because
    the branches are not equally hard. The compressed chain delocalizes
    across all n atoms and a window of ell misses more of that, while
    the stretched chain is nearly decoupled and a window of the same
    ell captures almost everything. Measured at n=6, the compressed
    bracket is 3-4.3x the stretched one at equal ell, so the budget
    buys width where width is scarce. A single shared knob cannot say
    that."""
    ells = tuple(range(2, min(n - 1, ell_max) + 1))

    def branch(d):
        return lambda ell, _up: h_chain_bracket(n, d, ell, correction_iters)

    stages = (Stage("compressed", ells, branch(d_near), lambda e: 4.0 ** e),
              Stage("stretched", ells, branch(d_far), lambda e: 4.0 ** e))
    return compose("h-chain-gap", tol, stages,
                   lambda c: c["stretched"] - c["compressed"],
                   context=f"n={n} d={d_near:g}->{d_far:g}")


def keff_dispatch(rx: dict, tol_pcm: float, m_max: int = 40,
                  jump: bool = True) -> Certified:
    """k_eff for a reactor, certified to a tolerance stated the way the
    field states it, in pcm, meaning hundred-thousandths of k. The knob
    is the number of fission-source iterations spent sharpening the
    trial flux. It is a ladder the planner can trust to be monotone,
    because the bracket provably NESTS. If A phi >= m phi then applying
    A to both sides preserves the inequality, since A >= 0, so the
    lower bound can only rise and the upper only fall. Nothing here can
    get worse by working harder.

    The ladder ends at a floor, not at a wall. Past some rung the
    sandwich is thinner than the certified error of the linear solve
    inside it, and further iterations buy nothing. The refusal says so,
    and prices the next move as precision rather than effort. Measured
    on the default slab that floor is around 10^-6 pcm. That is a
    million times below the 1 pcm the field argues about, and a million
    times below the discretization error that separates this model from
    a reactor. On this problem the certificate is never the weak link,
    which is worth saying out loud, because a bound whose own limit
    nobody has looked for is a bound nobody knows the strength of.

    One rewrite, one knob, and deliberately no competitor. The obvious
    rival is to hand the matrix to a dense eigensolver. That is not a
    rewrite this planner can race, because it returns an eigenvalue
    without a bound, and no cheap bound exists for a non-symmetric
    one.
    An entry with no certificate is not a cheaper answer to the same
    question. It is an answer to a different one."""
    from scipy.linalg import lu_factor, lu_solve
    L, F = rx["L"], rx["F"]
    u = mmatrix_witness(L)
    lu = lu_factor(L)
    knobs = tuple(range(2, m_max + 1, 2))

    def run(m):
        # ponytail: each rung restarts from a flat flux instead of
        # resuming a cached iterate, so the declared cost (m solves)
        # is the cost actually measured. The whole ladder is O(m^2)
        # back-substitutions on a few hundred unknowns -- milliseconds.
        phi = np.ones(len(L))
        for _ in range(m):
            phi = lu_solve(lu, F @ phi)
            phi /= phi.max()
        return keff_bracket(L, F, phi, u)

    def beyond():
        return (f"the ladder ends at {knobs[-1]} iterations; past its "
                "floor the bracket is limited by the certified residual "
                "of the linear solve, so the next move is extended "
                "precision or a better-conditioned discretization, not "
                "more iterations")

    rw = Rewrite("power", knobs, float, run, beyond)
    return plan("criticality", tol_pcm * 1e-5, [rw], jump=jump,
                context=rx["label"])


def keff_continuum_bracket(width: float = 70.0, Ns=(25, 50, 100, 200),
                           tol_pcm: float = 0.01, **xs) -> Certified:
    """The continuum answer, and the price of asking for it. Everything
    keff_bracket certifies is about a reactor made of finitely many
    cells. The reactor itself has none, and the distance between the
    two is the model's error, not the certificate's. This closes that
    gap the only way it can be closed cheaply. Refine the mesh, watch
    the answer move, and bound where it is going.

    Each rung is a rigorously certified discrete bracket, and the
    rungs are then fed to the same grid-convergence machinery the SPH
    sea-wall uses. Two errors compose. The distance from the finest
    mesh to h -> 0 is measured, not proven, so it arrives EMPIRICAL and
    the composed answer inherits that. The finest rung's own half-width
    is rigorous and simply adds. One precondition is checked rather
    than hoped for. The discrete brackets must be far narrower than the
    differences between rungs, or the measured order is reading its own
    noise. Measured on the default ladder they are 0.7% of the
    smallest difference.

    The result is worth looking at closely. The continuum certificate
    is hundreds of times WIDER than the discrete one it is built from,
    and a tier weaker. That is not a defect in either. It is the honest
    exchange rate between a rigorous statement about a model and a
    statistical one about the world, and it is why this library prints
    both rather than quietly reporting the tighter number."""
    rungs = [keff_dispatch(slab_reactor(N=N, width=width, **xs), tol_pcm)
             for N in Ns]
    return continuum_limit(rungs, [width / N for N in Ns],
                           "continuum k_eff: mesh ladder "
                           f"N={'/'.join(map(str, Ns))}", 1e5, " pcm")


def junction_dispatch(dev: dict, volts: float, tol_nC: float,
                      m_max: int = 80, jump: bool = True) -> Certified:
    """The depletion charge of a junction, certified to a tolerance
    stated in the unit a process engineer reads. That unit is nC/cm^2
    of charge per unit area, the quantity a capacitance-voltage
    measurement integrates. The knob is the number of Newton steps.

    This ladder has a shape none of the others has. Everywhere else in
    this library the error shrinks smoothly and the planner walks down
    it. Here the early rungs return NOTHING, not even a loose bound,
    because Kantorovich either closes or does not, and below the
    critical rung it does not. Then Newton's quadratic convergence
    squares the residual each step, and the certificate appears at full
    strength in one rung.

    The measured numbers on the default junction at 1 V are worth
    quoting, because they are the argument for doing any of this. The
    last refusing rung has residual 1.3 * 10^-3. One Newton step later
    the residual is 5 * 10^-6 and the certificate closes, at 0.78
    nC/cm^2, which is half a per cent of the answer. Five parts in a
    million of residual buys half a per cent of charge. The gap is
    the inverse Jacobian, about 236 here, times the charge
    functional's own gradient, and neither is visible in the residual.
    A stopping rule written on the residual is guessing at that
    product. One more step, and the certified error falls to 10^-5
    nC/cm^2.

    Why the discrete solution is unique is worth saying, since
    Kantorovich alone would only give uniqueness in a ball. The
    nonlinearity exp(psi - V) - exp(-psi) is strictly increasing, so
    the Jacobian is a Z-matrix with a strictly dominant diagonal
    everywhere, not just at the answer. That makes it a nonsingular
    M-matrix at every potential. A strictly monotone operator has one
    root. The enclosure is around the solution, not around a
    solution."""
    knobs = tuple(range(1, m_max + 1))

    def run(m):
        # ponytail: each rung restarts from the charge-neutral guess
        # rather than resuming the previous rung's iterate, so the
        # declared cost -- m linear solves -- is the cost actually
        # paid. The whole ladder is milliseconds on a few hundred
        # unknowns.
        return junction_charge_bracket(dev, volts,
                                       junction_potential(dev, volts, m))

    def beyond():
        return (f"the ladder ends at {knobs[-1]} Newton steps; if the "
                "certificate has not closed by then the damping cap is "
                "the binding constraint, not the tolerance, since the "
                f"potential must travel {abs(volts) / _VT:.0f} thermal "
                "volts and each step may move it by at most three")

    rw = Rewrite("newton", knobs, float, run, beyond)
    return plan("depletion-charge", tol_nC, [rw], jump=jump,
                context=f"{dev['label']} at {volts:g} V")


def junction_continuum_charge(volts: float = 1.0, Ns=(50, 100, 200, 400),
                              tol_nC: float = 1e-3, length_um: float = 1.0,
                              Na: float = 1e17, Nd: float = 1e17,
                              **kw) -> Certified:
    """The charge of the junction rather than of its mesh. Kantorovich
    proves an exact solution of the DISCRETE equations exists and
    encloses it; nothing in that argument knows the cells are there.
    Measured on this device the gap is not a rounding detail: the
    discrete certificate is worth 10^-8 nC/cm^2 and the mesh is worth
    10^-2, six orders apart, so the number a reader should distrust is
    the one the rigorous half is silent about.

    Refine, watch, and bound where it is going. Each rung is a
    certified discrete answer from junction_dispatch and the ladder is
    read by the same grid-convergence machinery the reactor and the
    sea wall use, through continuum_limit.

    One thing was measured and NOT built, which is worth recording
    because it looks like an obvious composed plan. Splitting a single
    tolerance between the mesh and the Newton iteration derives
    nothing, because the Newton half is free: at N=200 the certificate
    goes from 0.78 nC/cm^2 -- thirty-three times too coarse to read
    the mesh trend at all -- to 1.1*10^-5 in ONE step, two thousand
    times tighter than the ladder needs. Every budget that matters
    picks the same rung, so there is no exchange rate to price and a
    stage split would be decoration. Quadratic convergence is what
    does it, and it is the same fact that makes this ladder's rungs
    cheap enough to climb."""
    rungs = [junction_dispatch(pn_junction(N=N, length_um=length_um, Na=Na,
                                           Nd=Nd), volts, tol_nC, **kw)
             for N in Ns]
    return continuum_limit(rungs, [1.0 / N for N in Ns],
                           "continuum depletion charge: mesh ladder "
                           f"N={'/'.join(map(str, Ns))}",
                           1.0, " nC/cm^2")
