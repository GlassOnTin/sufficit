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
    err = A / (p + 1) * rho ** (p + 1) / (1 - rho)
    return Certified(float(total.real), err, Tier.RIGOROUS,
                     (f"multipole p={p} rho={rho:.3g}",))


def multipole_to_tol(q: np.ndarray, src: np.ndarray, center: complex,
                     z: complex, tol: float) -> Certified:
    """Inverse rewrite: minimal truncation order whose tail bound meets tol.
    Closed-form sufficient order from the geometric part, then walked down."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    A, rho = _far_geometry(q, src, center, z)
    p = 0
    if A > 0 and rho > 0:
        t = tol * (1 - rho) / A
        if t < 1:
            p = max(0, math.ceil(math.log(t) / math.log(rho)) - 1)
        def tail(p):
            return A / (p + 1) * rho ** (p + 1) / (1 - rho)
        while p > 0 and tail(p - 1) <= tol:
            p -= 1
    return multipole_far_potential(q, src, center, z, p)
