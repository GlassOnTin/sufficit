"""Sufficit Phase 0: the error-algebra IR, minimal seed.

A Certified value = (value, err, tier, provenance). err bounds the distance to
the true value (abs for scalars, 2-norm for vectors) and composes through
rewrites the way derivatives compose through autodiff. Tiers degrade to the
weakest input; provenance records which rewrites produced the bound.

Bounds are exact-arithmetic: floating-point rounding is not yet carried.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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

    def _combine(self, other: "Certified", value, err, note: str) -> "Certified":
        return Certified(value, err, min(self.tier, other.tier),
                         self.provenance + other.provenance + (note,))

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
    return Certified(f(c.value), L * c.err, c.tier, c.provenance + (note,))


def lowrank_matvec(K: np.ndarray, q: np.ndarray, rank: int) -> Certified:
    """Rewrite: compress K@q to numerical rank. ||K - K_r||_2 = sigma_{r+1},
    so ||K_r q - K q||_2 <= sigma_{r+1} ||q||_2. Rigorous (exact arithmetic).
    """
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
    value = (U[:, :rank] * s[:rank]) @ (Vt[:rank] @ q)
    err = float(s[rank] * np.linalg.norm(q)) if rank < len(s) else 0.0
    return Certified(value, err, Tier.RIGOROUS,
                     (f"lowrank r={rank} sigma={err:.3g}",))


def multipole_far_potential(q: np.ndarray, src: np.ndarray, center: complex,
                            z: complex, p: int) -> Certified:
    """Rewrite: truncate the 2D multipole expansion of sum_j q_j log|z - src_j|
    at order p (Greengard-Rokhlin tail bound). Requires |z - center| strictly
    outside the source radius."""
    dz = src - center
    r = float(np.max(np.abs(dz)))
    w = z - center
    d = abs(w)
    if d <= r:
        raise ValueError(f"target distance {d:.3g} <= source radius {r:.3g}: "
                         "multipole tail bound invalid")
    Q = float(np.sum(q))
    total = Q * np.log(w)
    for k in range(1, p + 1):
        total += (-np.sum(q * dz**k) / k) / w**k
    A = float(np.sum(np.abs(q)))
    rho = r / d
    err = A / (p + 1) * rho ** (p + 1) / (1 - rho)
    return Certified(float(total.real), err, Tier.RIGOROUS,
                     (f"multipole p={p} rho={rho:.3g}",))
