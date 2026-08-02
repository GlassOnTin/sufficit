"""Phase 0 acceptance tests: every claimed bound must contain the actual error,
measured against brute force. Bounds are exact-arithmetic; tests allow 1e-9
relative slack for floating point, which the IR does not yet carry."""
import math
import random
import re

import numpy as np
import pytest

import sufficit as sf


def _knob(c, key):
    """Read a chosen parameter (rank, order, rounds) back out of provenance."""
    return int(re.search(rf"{key}=(\d+)", c.provenance[0]).group(1))


def test_tier_degrades_to_weakest():
    a = sf.Certified(1.0, 0.1, sf.Tier.RIGOROUS, ("a",))
    b = sf.Certified(2.0, 0.1, sf.Tier.EMPIRICAL, ("b",))
    assert (a + b).tier == sf.Tier.EMPIRICAL
    assert (a * b).tier == sf.Tier.EMPIRICAL
    assert (a - b).tier == sf.Tier.EMPIRICAL


def test_provenance_accumulates():
    a = sf.Certified(1.0, 0.1, sf.Tier.RIGOROUS, ("rank-probe",))
    b = sf.Certified(2.0, 0.1, sf.Tier.RIGOROUS, ("multipole p=8",))
    assert (a + b).provenance == ("rank-probe", "multipole p=8", "add")


def test_arithmetic_bounds_contain_true_error():
    rng = random.Random(0)
    for _ in range(2000):
        t1, t2 = rng.uniform(-10, 10), rng.uniform(-10, 10)
        e1, e2 = rng.uniform(0, 1), rng.uniform(0, 1)
        a = sf.Certified(t1 + rng.uniform(-e1, e1), e1, sf.Tier.RIGOROUS, ())
        b = sf.Certified(t2 + rng.uniform(-e2, e2), e2, sf.Tier.RIGOROUS, ())
        for op, true in ((a + b, t1 + t2), (a - b, t1 - t2), (a * b, t1 * t2)):
            assert abs(op.value - true) <= op.err * (1 + 1e-9) + 1e-12


def test_lipschitz_apply():
    # sin is 1-Lipschitz; true input 2.0 declared with err 0.3
    c = sf.Certified(2.25, 0.3, sf.Tier.RIGOROUS, ("input",))
    out = sf.lipschitz(math.sin, 1.0, c, "sin")
    assert abs(out.value - math.sin(2.0)) <= out.err
    assert out.err == pytest.approx(0.3)


def test_lowrank_matvec_bound_holds():
    rng = np.random.default_rng(1)
    # smooth kernel => fast singular-value decay
    x, y = rng.uniform(0, 1, 60), rng.uniform(3, 4, 50)
    K = 1.0 / np.abs(x[:, None] - y[None, :])
    q = rng.standard_normal(50)
    exact = K @ q
    for rank in (1, 3, 6, 10):
        c = sf.lowrank_matvec(K, q, rank)
        actual = np.linalg.norm(c.value - exact)
        assert actual <= c.err * (1 + 1e-9) + 1e-12, (rank, actual, c.err)
    # bound must be meaningful, not vacuous: rank 10 of a smooth kernel is tight
    assert sf.lowrank_matvec(K, q, 10).err < 1e-6 * np.linalg.norm(exact)


def test_multipole_bound_holds():
    rng = np.random.default_rng(2)
    src = (rng.uniform(-0.5, 0.5, 40) + 1j * rng.uniform(-0.5, 0.5, 40))
    q = rng.uniform(-1, 1, 40)
    z = 3.0 + 2.0j
    exact = float(np.sum(q * np.log(np.abs(z - src))))
    prev = math.inf
    for p in (2, 5, 10, 20):
        c = sf.multipole_far_potential(q, src, 0.0 + 0.0j, z, p)
        assert abs(c.value - exact) <= c.err * (1 + 1e-9) + 1e-12
        assert c.err < prev  # bound tightens with order
        prev = c.err


def test_multipole_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        # target inside the source radius: the tail bound is invalid
        sf.multipole_far_potential(
            np.array([1.0]), np.array([0.9 + 0.0j]), 0.0 + 0.0j, 0.5 + 0.0j, 5
        )


def test_fail_p_composes_by_union_bound():
    a = sf.Certified(1.0, 0.1, sf.Tier.RIGOROUS, (), fail_p=1e-3)
    b = sf.Certified(2.0, 0.1, sf.Tier.RIGOROUS, (), fail_p=1e-4)
    assert (a + b).fail_p == pytest.approx(1.1e-3)
    assert (a * b).fail_p == pytest.approx(1.1e-3)
    assert sf.lipschitz(math.sin, 1.0, a, "sin").fail_p == 1e-3
    big = sf.Certified(0.0, 0.0, sf.Tier.RIGOROUS, (), fail_p=0.7)
    assert (big + big).fail_p == 1.0  # capped


def test_randomized_probe_bound_holds_over_trials():
    """Per-trial fail_p is 1e-10, so 300 trials must show zero violations."""
    rng = np.random.default_rng(4)
    overshoots = []
    for _ in range(300):
        x, y = rng.uniform(0, 1, 50), rng.uniform(2, 3, 40)
        K = 1.0 / np.abs(x[:, None] - y[None, :])
        q = rng.standard_normal(40)
        c = sf.randomized_lowrank_matvec(K, q, 5, n_probes=10, rng=rng)
        actual = np.linalg.norm(c.value - K @ q)
        assert actual <= c.err * (1 + 1e-9) + 1e-12
        assert c.fail_p == pytest.approx(1e-10)
        overshoots.append(c.err / max(actual, 1e-300))
    # non-vacuous: the estimator overshoots by a bounded factor, not orders
    # of magnitude beyond its ~10 sqrt(2/pi) sqrt(m) worst case
    assert np.median(overshoots) < 1e4


def test_end_to_end_chain_with_randomized_probe():
    """Chain a probabilistic certificate through the algebra: the composed
    bound must hold and the stated failure probability must survive."""
    rng = np.random.default_rng(5)
    tx = rng.uniform(0, 1, 80)
    near_y = rng.uniform(2, 3, 60)
    qn = rng.uniform(-1, 1, 60)
    K = np.log(np.abs(tx[:, None] - near_y[None, :]))

    near = sf.randomized_lowrank_matvec(K, qn, 8, n_probes=10, rng=rng)
    obs = sf.lipschitz(np.mean, 1.0 / math.sqrt(len(tx)), near, "mean")
    truth = float(np.mean(K @ qn))
    assert abs(obs.value - truth) <= obs.err * (1 + 1e-9) + 1e-12
    assert obs.tier == sf.Tier.RIGOROUS
    assert obs.fail_p == pytest.approx(1e-10)


def test_svd_to_tol_meets_and_scales():
    rng = np.random.default_rng(6)
    x, y = rng.uniform(0, 1, 60), rng.uniform(2, 3, 50)
    K = 1.0 / np.abs(x[:, None] - y[None, :])
    q = rng.standard_normal(50)
    exact = K @ q
    ranks = []
    for tol in (1e-2, 1e-4, 1e-6, 1e-8):
        c = sf.lowrank_matvec_to_tol(K, q, tol)
        assert c.err <= tol
        assert np.linalg.norm(c.value - exact) <= tol * (1 + 1e-9) + 1e-12
        ranks.append(_knob(c, "r"))
    assert ranks == sorted(ranks) and ranks[-1] > ranks[0]


def test_randomized_to_tol_meets_and_accounts_failure():
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 1, 60), rng.uniform(2, 3, 50)
    K = 1.0 / np.abs(x[:, None] - y[None, :])
    q = rng.standard_normal(50)
    ranks = []
    for tol in (1e-1, 1e-4, 1e-7):
        c = sf.randomized_lowrank_matvec_to_tol(K, q, tol, rng=rng)
        assert c.err <= tol
        assert np.linalg.norm(c.value - K @ q) <= tol * (1 + 1e-9) + 1e-12
        # fail_p must account for every probe round the search consumed
        assert c.fail_p == pytest.approx(_knob(c, "rounds") * 1e-10)
        ranks.append(_knob(c, "r"))
    assert ranks == sorted(ranks) and ranks[-1] > ranks[0]


def test_randomized_to_tol_refuses_impossible_tolerance():
    rng = np.random.default_rng(8)
    K = rng.standard_normal((20, 15))
    q = rng.standard_normal(15)
    with pytest.raises(ValueError):
        sf.randomized_lowrank_matvec_to_tol(K, q, 1e-300, rng=rng)


def test_multipole_to_tol_minimal_order():
    rng = np.random.default_rng(2)
    src = rng.uniform(-0.5, 0.5, 40) + 1j * rng.uniform(-0.5, 0.5, 40)
    q = rng.uniform(-1, 1, 40)
    z = 3.0 + 2.0j
    exact = float(np.sum(q * np.log(np.abs(z - src))))
    orders = []
    for tol in (1e-1, 1e-4, 1e-8, 1e-12):
        c = sf.multipole_to_tol(q, src, 0j, z, tol)
        assert c.err <= tol
        assert abs(c.value - exact) <= tol * (1 + 1e-9) + 1e-12
        p = _knob(c, "p")
        orders.append(p)
        if p > 0:  # minimality: one order less must miss the tolerance
            assert sf.multipole_far_potential(q, src, 0j, z, p - 1).err > tol
    assert orders == sorted(orders) and orders[-1] > orders[0]
    with pytest.raises(ValueError):
        sf.multipole_to_tol(q, src, 0j, z, 0.0)


def test_tolerance_driven_end_to_end():
    """The query-first interface: state eps, the system picks every knob.
    Certified and actual error meet eps; cost grows only as eps shrinks."""
    rng = np.random.default_rng(9)
    tx = rng.uniform(0, 1, 80)
    near_y = rng.uniform(2, 3, 60)
    far_src = 10.0 + rng.uniform(-0.5, 0.5, 50) + 1j * rng.uniform(-0.5, 0.5, 50)
    qn, qf = rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 50)
    K = np.log(np.abs(tx[:, None] - near_y[None, :]))
    probe = 0.5 + 0.0j
    truth = float(np.mean(K @ qn)) + float(np.sum(qf * np.log(np.abs(probe - far_src))))

    costs = []
    for eps in (1e-3, 1e-6, 1e-9):
        # equal split; the near stage's budget is pre-scaled by the mean's
        # inverse Lipschitz constant sqrt(n)
        near = sf.lowrank_matvec_to_tol(K, qn, eps / 2 * math.sqrt(len(tx)))
        obs = sf.lipschitz(np.mean, 1.0 / math.sqrt(len(tx)), near, "mean")
        far = sf.multipole_to_tol(qf, far_src, 10.0 + 0.0j, probe, eps / 2)
        ans = obs + far
        assert ans.err <= eps
        assert abs(ans.value - truth) <= eps * (1 + 1e-9) + 1e-12
        assert ans.tier == sf.Tier.RIGOROUS
        costs.append((_knob(near, "r"), _knob(far, "p")))
    assert costs == sorted(costs) and costs[-1] > costs[0]


def test_treecode_certified_pointwise_and_cheaper():
    """Phase 1: hierarchical evaluation must certify every target to eps and
    cost fewer kernel-equivalent ops than the dense sum."""
    rng = np.random.default_rng(10)
    src = rng.uniform(0, 1, 2000) + 1j * rng.uniform(0, 1, 2000)
    tgt = rng.uniform(0, 1, 2000) + 1j * rng.uniform(0, 1, 2000)
    q = rng.uniform(-1, 1, 2000)
    dense = np.log(np.abs(tgt[:, None] - src[None, :])) @ q
    prev_ops = 0
    for eps in (1e-3, 1e-6, 1e-9):
        c, stats = sf.treecode_potential(tgt, src, q, eps)
        # the pointwise guarantee, against brute force
        assert np.max(np.abs(c.value - dense)) <= eps * (1 + 1e-9) + 1e-12
        assert stats["max_bound"] <= eps * (1 + 1e-9)
        # the vector certificate composes with the rest of the algebra
        assert np.linalg.norm(c.value - dense) <= c.err * (1 + 1e-9) + 1e-12
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
        # cost scales with precision, and beats dense at every tolerance
        assert stats["ops"] > prev_ops
        assert stats["ops"] < stats["dense_ops"]
        prev_ops = stats["ops"]


def test_treecode_speedup_grows_with_n():
    """O(N log N) vs O(N^2): the op-count advantage must widen with N."""
    rng = np.random.default_rng(11)
    speedups = []
    for n in (500, 2000, 8000):
        src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
        tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
        q = rng.uniform(-1, 1, n)
        _, stats = sf.treecode_potential(tgt, src, q, 1e-6)
        speedups.append(stats["dense_ops"] / stats["ops"])
    assert speedups == sorted(speedups) and speedups[-1] > speedups[0]


def test_fmm_certified_pointwise():
    """M2L + local expansions must keep the pointwise certificate."""
    rng = np.random.default_rng(12)
    src = rng.uniform(0, 1, 2000) + 1j * rng.uniform(0, 1, 2000)
    tgt = rng.uniform(0, 1, 2000) + 1j * rng.uniform(0, 1, 2000)
    q = rng.uniform(-1, 1, 2000)
    dense = np.log(np.abs(tgt[:, None] - src[None, :])) @ q
    for eps in (1e-3, 1e-6, 1e-9):
        c, stats = sf.fmm_potential(tgt, src, q, eps)
        assert np.max(np.abs(c.value - dense)) <= eps * (1 + 1e-9) + 1e-12
        assert stats["max_bound"] <= eps * (1 + 1e-9)
        assert np.linalg.norm(c.value - dense) <= c.err * (1 + 1e-9) + 1e-12
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
        assert stats["ops"] < stats["dense_ops"]
        assert stats["m2l_pairs"] > 0  # M2L path actually exercised


def test_m2m_translation_is_exact():
    """2D M2M is lower-triangular: coefficients built by recursive child
    translation must equal direct P2M coefficients to FP precision, so M2M
    adds no error term to the certificate."""
    rng = np.random.default_rng(14)
    src = rng.uniform(0, 1, 500) + 1j * rng.uniform(0, 1, 500)
    q = rng.uniform(-1, 1, 500)
    p = 25   # Pascal rows stay exact in float64 well beyond this
    ops = {"p2m": 0, "m2m": 0}
    root = sf._root(src, 10)
    sf._ensure_coeffs(root, src, q, p, ops, via_m2m=True)
    assert ops["m2m"] > 0 and root.children      # M2M path actually taken
    direct = sf._root(src, 10)
    sf._ensure_coeffs(direct, src, q, p, {"p2m": 0}, via_m2m=False)
    A, r = float(np.abs(q).sum()), root.half * math.sqrt(2)
    scale = A * r ** np.arange(1, p + 1)
    assert np.all(np.abs(root.coeffs - direct.coeffs) <= 1e-12 * scale)


def test_fmm_upward_pass_uses_m2m():
    """P2M must now touch sources only at leaves; internal cells translate."""
    rng = np.random.default_rng(15)
    n = 20000
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    q = rng.uniform(-1, 1, n)
    _, stats = sf.fmm_potential(tgt, src, q, 1e-6)
    assert stats["m2m"] > 0
    # leaf-only P2M is O(N * max order); the old per-cell scheme cost
    # 2,836,416 ops on this exact input
    assert stats["p2m"] + stats["m2m"] < 2_836_416


def test_fmm_beats_treecode_at_scale():
    """The point of M2L: strictly fewer ops than the treecode as N grows."""
    rng = np.random.default_rng(13)
    n = 10000
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    q = rng.uniform(-1, 1, n)
    _, tree_stats = sf.treecode_potential(tgt, src, q, 1e-6)
    _, fmm_stats = sf.fmm_potential(tgt, src, q, 1e-6)
    assert fmm_stats["ops"] < tree_stats["ops"]


def test_blackbox_hmatrix_certified_and_reusable():
    """Black-box kernel: build once at ~N^2 kernel evals, then every apply
    is cheap and carries a certificate valid for any charge vector, with
    pointwise error <= eps * ||q||_2."""
    rng = np.random.default_rng(16)
    n = 4000
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    kernel = lambda t, s: 1.0 / np.sqrt(np.abs(t[:, None] - s[None, :]) ** 2
                                        + 0.01)          # Plummer, no analytic
    eps = 1e-6                                           # structure exposed
    plan = sf.BlackboxHMatrix(kernel, tgt, src, eps, leaf_size=96, rng=rng)
    assert plan.fail_p > 0                # probabilistic certificate, stated
    assert plan.stats["kernel_evals"] <= n * n
    dense = kernel(tgt, src)
    for _ in range(2):                    # one plan, many charge vectors
        q = rng.standard_normal(n)
        c, stats = plan.apply(q)
        assert np.max(np.abs(c.value - dense @ q)) \
            <= eps * np.linalg.norm(q) * (1 + 1e-9)
        assert stats["max_bound"] <= eps * np.linalg.norm(q) * (1 + 1e-9)
        assert np.linalg.norm(c.value - dense @ q) <= c.err * (1 + 1e-9) + 1e-12
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == plan.fail_p
        assert stats["apply_flops"] < 0.5 * n * n


def test_blackbox_agrees_with_analytic_fmm():
    """Mutual certification: the black-box compression of the log kernel and
    the analytic FMM must agree within the sum of their bounds."""
    rng = np.random.default_rng(17)
    n = 1500
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    q = rng.uniform(-1, 1, n)
    eps = 1e-6
    kernel = lambda t, s: np.log(np.abs(t[:, None] - s[None, :]))
    bb, bb_stats = sf.BlackboxHMatrix(kernel, tgt, src, eps, rng=rng).apply(q)
    an, _ = sf.fmm_potential(tgt, src, q, eps)
    assert np.max(np.abs(bb.value - an.value)) \
        <= bb_stats["max_bound"] + eps + 1e-12


def _ising_torus_logZ_density_tm(m, betaJ):
    """Exact log Z / N for the m x m periodic Ising lattice, weight
    exp(betaJ * sum s_i s_j), via row transfer matrix. Certificates are
    scoped to m >= 12; m=10 truth differs from that by wrap terms
    ~2 t^10/m per site — orders below every bound asserted against it,
    and 2^12-state matrices would slow the suite 10x."""
    s = np.arange(2 ** m, dtype=np.uint64)
    rot = (s >> np.uint64(1)) | ((s & np.uint64(1)) << np.uint64(m - 1))
    intra = m - 2 * np.bitwise_count(s ^ rot).astype(float)
    inter = m - 2 * np.bitwise_count(s[:, None] ^ s[None, :]).astype(float)
    A = np.exp(betaJ * intra)[:, None] * np.exp(betaJ * inter)
    return math.log(np.trace(np.linalg.matrix_power(A, m))) / (m * m)


def test_ising_transfer_matrix_matches_exhaustive():
    """Validate the truth generator itself on the 4x4 torus (2^16 states)."""
    m = 4
    bits = (np.arange(2 ** 16)[:, None] >> np.arange(16)[None, :]) & 1
    spins = 1 - 2 * bits.astype(float)          # site (x, y) -> bit 4*x + y
    energy = np.zeros(2 ** 16)
    for x in range(m):
        for y in range(m):
            i = 4 * x + y
            energy += spins[:, i] * (spins[:, 4 * ((x + 1) % m) + y]
                                     + spins[:, 4 * x + (y + 1) % m])
    for bJ in (0.005, 0.3):
        exact = math.log(np.sum(np.exp(bJ * energy))) / 16
        assert abs(_ising_torus_logZ_density_tm(m, bJ) - exact) < 1e-12


def test_ising_cycle_enumeration():
    """Anchored simple cycles on Z^2: one square and two dominoes below 8
    edges; 7 octagons (perimeter-8 polyomino boundaries) at 8; 28 at 10."""
    assert sorted(sf._ising2d_anchored_cycles(7)) == [4, 6, 6]
    from collections import Counter
    counts = Counter(sf._ising2d_anchored_cycles(10))
    assert counts == {4: 1, 6: 2, 8: 7, 10: 28}


def test_ising_polymer_shapes_and_series_coefficients():
    """Union polymers and Ursell pairs: the per-site coefficients must
    reproduce the known 2D Ising high-temperature series
    log Z/N = log 2 + 2 log cosh + t^4 + 2 t^6 + (9/2) t^8 + 12 t^10."""
    sizes = sorted(len(E) for E, _ in sf._ising2d_polymer_shapes(10))
    # 38 cycles plus 2 figure-eights (4+4) and 8 square-domino unions (4+6)
    assert sizes.count(8) == 7 + 2 and sizes.count(10) == 28 + 8
    assert sf._ising2d_logz_coeffs() == {4: 1.0, 6: 2.0, 8: 4.5, 10: 12.0}


def test_ising_series_matches_transfer_matrix_tightly():
    """With coefficients exact through t^10, the truncated series must
    agree with the exact torus to ~t^12 + wrap level — far below the KP
    certificate, so this catches coefficient bugs containment cannot.
    (m=10 wrap contamination is ~2 t^10/m per site, below thresholds.)"""
    for bJ, thresh in ((0.05, 5e-12), (0.08, 2e-11)):
        diff = abs(sf.ising2d_logZ_density(bJ).value
                   - _ising_torus_logZ_density_tm(10, bJ))
        assert diff < thresh, (bJ, diff)


def test_ising_cluster_expansion_certified():
    """Phase 2: certified free-energy density inside the high-temperature
    region, verified against the exact 10x10 transfer matrix (the bound
    covers every m x m torus with m >= 8)."""
    for bJ in (0.001, 0.005, 0.012):
        truth = _ising_torus_logZ_density_tm(10, bJ)
        c = sf.ising2d_logZ_density(bJ)
        assert abs(c.value - truth) <= c.err * (1 + 1e-9) + 1e-15
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
    # deep inside the region the certificate is tight enough to be useful
    c = sf.ising2d_logZ_density(0.001, tol=1e-9)
    assert c.err <= 1e-9
    # beta = 0: exact in exact arithmetic, but computing log 2 rounds
    c0 = sf.ising2d_logZ_density(0.0)
    assert c0.value == pytest.approx(math.log(2.0)) and c0.err <= 1e-15


def test_ising_refuses_outside_certified_region():
    """The first rewrite with a validity region: outside the certified
    convergence radius it must refuse, not extrapolate."""
    with pytest.raises(ValueError):
        sf.ising2d_logZ_density(0.1)          # beyond the certified radius
    with pytest.raises(ValueError):
        sf.ising2d_logZ_density(0.012, tol=1e-12)   # tol unreachable at L=8


def _ising_torus_bond_corr_tm(m, betaJ):
    """Exact <s_(0,0) s_(0,1)> (same-row adjacent pair) on the m x m torus:
    spin-product insertion in the row transfer matrix."""
    s = np.arange(2 ** m, dtype=np.uint64)
    rot = (s >> np.uint64(1)) | ((s & np.uint64(1)) << np.uint64(m - 1))
    intra = m - 2 * np.bitwise_count(s ^ rot).astype(float)
    inter = m - 2 * np.bitwise_count(s[:, None] ^ s[None, :]).astype(float)
    A = np.exp(betaJ * intra)[:, None] * np.exp(betaJ * inter)
    Am = np.linalg.matrix_power(A, m)
    ins = (1 - 2 * (s & np.uint64(1)).astype(float)) \
        * (1 - 2 * ((s >> np.uint64(1)) & np.uint64(1)).astype(float))
    return float(np.sum(ins * np.diag(Am)) / np.trace(Am))


def test_ising_tm_correlation_matches_exhaustive():
    m = 4
    bits = (np.arange(2 ** 16)[:, None] >> np.arange(16)[None, :]) & 1
    spins = 1 - 2 * bits.astype(float)          # site (x, y) -> bit 4*x + y
    energy = np.zeros(2 ** 16)
    for x in range(m):
        for y in range(m):
            energy += spins[:, 4 * x + y] * (spins[:, 4 * ((x + 1) % m) + y]
                                             + spins[:, 4 * x + (y + 1) % m])
    for bJ in (0.005, 0.3):
        w = np.exp(bJ * energy)
        exact = float(np.sum(spins[:, 0] * spins[:, 1] * w) / np.sum(w))
        assert abs(_ising_torus_bond_corr_tm(m, bJ) - exact) < 1e-12


def test_pinned_enumeration_vs_bruteforce():
    """The pinned-subgraph enumerator against brute force over all small
    edge sets: same subgraphs, no duplicates."""
    import itertools
    a, b = (0, 0), (1, 0)
    got = sf._connected_pinned_subgraphs(a, b, 3)
    assert len(got) == len(set(got))            # each exactly once
    verts = [(x, y) for x in range(-3, 4) for y in range(-3, 4)]
    edges = sorted({tuple(sorted([v, (v[0] + dx, v[1] + dy)]))
                    for v in verts for dx, dy in ((1, 0), (0, 1))
                    if (v[0] + dx, v[1] + dy) in verts})
    expect = set()
    for n in (1, 2, 3):
        for combo in itertools.combinations(edges, n):
            deg = {}
            for u, v in combo:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            if {v for v, d in deg.items() if d % 2} != {a, b}:
                continue
            comp, todo = {combo[0][0]}, [combo[0][0]]
            while todo:
                x = todo.pop()
                for u, v in combo:
                    for p, o in ((u, v), (v, u)):
                        if p == x and o not in comp:
                            comp.add(o)
                            todo.append(o)
            if a in comp and all(u in comp for u, _ in combo):
                expect.add(frozenset(combo))
    assert set(got) == expect


def test_ising_bond_correlation_certified():
    """Local observable via pinned clusters, verified against the exact
    transfer matrix on the 10x10 torus."""
    for bJ in (0.001, 0.005, 0.012):
        truth = _ising_torus_bond_corr_tm(10, bJ)
        c = sf.ising2d_bond_correlation(bJ)
        assert abs(c.value - truth) <= c.err * (1 + 1e-9) + 1e-15
        assert c.value > 0 and c.tier == sf.Tier.RIGOROUS
    c = sf.ising2d_bond_correlation(0.005)
    assert c.err < 1e-4                          # useful, not just valid
    assert sf.ising2d_bond_correlation(0.0).value == 0.0


def test_ising_bond_correlation_refuses():
    with pytest.raises(ValueError):
        sf.ising2d_bond_correlation(0.1)


def test_interval_arithmetic_contains_truth():
    """Field ops checked against exact rationals, transcendentals against
    50-digit mpmath: the interval must always contain the true value."""
    from fractions import Fraction
    import mpmath
    mpmath.mp.dps = 50
    rng = random.Random(19)
    for _ in range(500):
        a, b = rng.uniform(-5, 5), rng.uniform(-5, 5)
        ia, ib = sf.Interval(a), sf.Interval(b)
        fa, fb = Fraction(a), Fraction(b)
        for iv, true in ((ia + ib, fa + fb), (ia - ib, fa - fb),
                         (ia * ib, fa * fb), (ia**3, fa**3)):
            assert Fraction(iv.lo) <= true <= Fraction(iv.hi)
        if b != 0:
            iv = ia / ib
            assert Fraction(iv.lo) <= fa / fb <= Fraction(iv.hi)
        for iv, true in ((ia.exp(), mpmath.exp(a)), (ia.tanh(), mpmath.tanh(a)),
                         (ia.cosh(), mpmath.cosh(a))):
            assert iv.lo <= true <= iv.hi
        if a > 0:
            iv = ia.log()
            assert iv.lo <= mpmath.log(a) <= iv.hi
    # cosh spanning zero attains its minimum 1 inside the interval
    iv = sf.Interval(-0.5, 0.3).cosh()
    assert iv.lo <= 1.0 <= iv.hi and iv.hi >= math.cosh(0.5)


def test_fp_error_now_carried():
    """The certificate can no longer dip below what float64 evaluation can
    support: the value interval's width enters the bound."""
    c = sf.ising2d_logZ_density(0.001)
    assert 1e-16 <= c.err <= 1e-12
    assert "+fp" in c.provenance[0]
    cb = sf.ising2d_bond_correlation(0.001)
    assert cb.err >= 1e-18 and "+fp" in cb.provenance[0]


def test_eulerian_counting_extends_region():
    """Eulerian encoding (4*3^(n-1) walks vs 16^n): the certified radius
    grows from tanh(betaJ) < 0.0167 to < ~0.085, and certificates in the
    newly covered territory still contain the exact answer."""
    c = sf.ising2d_logZ_density(0.05)              # refused before
    assert abs(c.value - _ising_torus_logZ_density_tm(10, 0.05)) <= c.err
    cb = sf.ising2d_bond_correlation(0.05)
    assert abs(cb.value - _ising_torus_bond_corr_tm(10, 0.05)) <= cb.err
    with pytest.raises(ValueError):
        sf.ising2d_logZ_density(0.086)             # new boundary enforced


def _gauss(x, sigma):
    return math.exp(-x * x / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))


def test_gaussian_laplace_formula():
    """The erfcx-based Laplace transform of the smearing kernel against
    direct numerical integration."""
    w = np.linspace(0, 60, 4_000_001)
    for omega, sigma, tau in ((1.0, 0.4, 3.0), (2.0, 0.2, 0.0), (0.6, 0.5, 14.0)):
        num = np.trapezoid(np.exp(-tau * w)
                           * np.exp(-(w - omega) ** 2 / (2 * sigma ** 2))
                           / (sigma * math.sqrt(2 * math.pi)), w)
        # rel limited by the trapezoid reference itself, not the formula
        assert sf._gauss_laplace(tau, omega, sigma) == pytest.approx(num, rel=1e-6)


def test_smeared_spectral_certified():
    """Phase 3: two-peak spectral density, exact correlator data in,
    certified smeared values out; the sup-bound c is checked against
    dense sampling of the actual kernel deviation."""
    E, a = (0.9, 1.9), (1.0, 0.7)
    ts = np.arange(1, 17)
    C = a[0] * np.exp(-E[0] * ts) + a[1] * np.exp(-E[1] * ts)
    for omega in (0.8, 1.2):
        for sigma in (0.5, 0.35, 0.25):
            truth = sum(ai * _gauss(omega - Ei, sigma) for ai, Ei in zip(a, E))
            c = sf.smeared_spectral(C, omega, sigma)
            assert abs(c.value - truth) <= c.err
            # non-vacuous everywhere; genuinely tight only at coarser
            # resolution — 15 exponentials against a sigma=0.25 Gaussian
            # in the e^w-weighted sup norm is a real resolution wall
            assert c.err < truth
            if sigma >= 0.35:
                assert c.err < 0.55 * truth
            assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
            assert "rho>=0" in c.provenance[0]


def test_smeared_resolution_costs_error():
    """The Phase 3 signature: sharper resolution (smaller sigma) is a more
    expensive question — the certified error must grow."""
    ts = np.arange(1, 17)
    C = np.exp(-0.9 * ts) + 0.7 * np.exp(-1.9 * ts)
    errs = [sf.smeared_spectral(C, 1.0, s).err for s in (0.6, 0.4, 0.25)]
    assert errs[0] < errs[1] < errs[2]


def test_smeared_sup_bound_is_a_sup_bound():
    """c must dominate the actual weighted kernel deviation everywhere."""
    ts = np.arange(1, 17)
    C = np.exp(-0.9 * ts) + 0.7 * np.exp(-1.9 * ts)
    g, cbound = sf._hlt_solve(len(C), 1.0, 0.35)
    w = np.linspace(0, 40, 800_001)
    krec = sum(gt * np.exp(-w * t) for gt, t in zip(g, range(2, len(C) + 1)))
    dev = np.abs(krec - np.exp(-(w - 1.0) ** 2 / (2 * 0.35 ** 2))
                 / (0.35 * math.sqrt(2 * math.pi))) * np.exp(w)
    assert np.max(dev) <= cbound * (1 + 1e-9)


def test_smeared_spectral_statistical():
    """Noisy data: the certificate degrades honestly to EMPIRICAL with a
    stated confidence, and still contains the truth."""
    rng = np.random.default_rng(20)
    E, a = (0.9, 1.9), (1.0, 0.7)
    ts = np.arange(1, 17)
    C = a[0] * np.exp(-E[0] * ts) + a[1] * np.exp(-E[1] * ts)
    cov = np.diag((1e-5 * C) ** 2)
    noisy = C + rng.standard_normal(len(C)) * 1e-5 * C
    truth = sum(ai * _gauss(1.0 - Ei, 0.35) for ai, Ei in zip(a, E))
    exact = sf.smeared_spectral(C, 1.0, 0.35)
    c = sf.smeared_spectral(noisy, 1.0, 0.35, cov=cov)
    assert abs(c.value - truth) <= c.err
    assert c.err > exact.err                     # noise costs certificate
    assert c.tier == sf.Tier.EMPIRICAL
    assert c.fail_p == pytest.approx(2 * math.erfc(5.0 / math.sqrt(2)))


def _mz_test_system(rng, fast_scale=1.0):
    """Dissipative slow-fast linear system: 2 slow observables weakly
    coupled to an 8-dimensional stiff fast sector."""
    A = np.zeros((10, 10))
    A[:2, :2] = [[-0.3, 0.2], [-0.2, -0.4]]
    A[:2, 2:] = 0.15 * rng.standard_normal((2, 8))
    A[2:, :2] = 0.15 * rng.standard_normal((8, 2))
    A[2:, 2:] = fast_scale * (-6.0 * np.eye(8) + 0.3 * rng.standard_normal((8, 8)))
    return A


def test_mz_linear_closure_certified():
    """Phase 4, rigorous tier: Markovian closure of a linear slow-fast
    system, verified against the exact full propagator."""
    from scipy.linalg import expm
    rng = np.random.default_rng(21)
    A = _mz_test_system(rng)
    x10 = np.array([1.0, -0.5])
    x0 = np.concatenate([x10, np.zeros(8)])
    for T in (1.0, 5.0, 20.0):
        exact = (expm(A * T) @ x0)[:2]
        c = sf.mz_closure_linear(A, 2, x0, T)
        assert np.linalg.norm(c.value - exact) <= c.err
        assert c.err < 0.5 * np.linalg.norm(x10)     # non-vacuous
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
        assert "mu=" in c.provenance[0]


def test_mz_bound_tightens_with_gap():
    """Physical provenance: the certificate depends on the fast sector's
    spectral gap, so a stiffer fast block must certify tighter."""
    rng = np.random.default_rng(22)
    x0 = np.concatenate([[1.0, -0.5], np.zeros(8)])
    base = sf.mz_closure_linear(_mz_test_system(rng, 1.0), 2, x0, 5.0)
    rng = np.random.default_rng(22)
    stiff = sf.mz_closure_linear(_mz_test_system(rng, 2.0), 2, x0, 5.0)
    assert stiff.err < base.err / 2


def test_mz_refuses_without_gap():
    """No spectral gap in the fast sector: no certified memory decay."""
    A = np.zeros((4, 4))
    A[:2, :2] = -np.eye(2)
    A[2:, 2:] = np.array([[0.1, 0.0], [0.0, -1.0]])  # unstable fast mode
    with pytest.raises(ValueError):
        sf.mz_closure_linear(A, 2, np.array([1.0, 0.0, 0.0, 0.0]), 1.0)


def test_mz_nonzero_fast_initial_condition():
    """The closure now covers arbitrary fast initial states: the decaying
    transient enters the bound instead of being assumed away."""
    from scipy.linalg import expm
    rng = np.random.default_rng(25)
    A = _mz_test_system(rng)
    x0 = np.concatenate([[1.0, -0.5], 0.3 * rng.standard_normal(8)])
    for T in (1.0, 5.0):
        exact = (expm(A * T) @ x0)[:2]
        c = sf.mz_closure_linear(A, 2, x0, T)
        assert np.linalg.norm(c.value - exact) <= c.err
        assert c.err < np.linalg.norm(x0)            # non-vacuous


def test_mz_slow_variable_search():
    """Automatic slow-variable identification: the designed slow pair is
    hidden by a random coordinate permutation; the search must recover
    it from the certificates alone."""
    from scipy.linalg import expm
    rng = np.random.default_rng(26)
    A = _mz_test_system(rng)
    x0 = np.concatenate([[1.0, -0.5], np.zeros(8)])
    perm = rng.permutation(10)
    Ap = A[np.ix_(perm, perm)]
    x0p = x0[perm]
    designed = {int(j) for j in range(10) if perm[j] < 2}

    # seeded with one target coordinate the discovery is exact
    tgt = min(designed)
    c, slow = sf.mz_search_slow(Ap, x0p, 5.0, targets=[tgt], tol=0.05)
    assert set(slow) == designed
    exact = (expm(Ap * 5.0) @ x0p)[np.array(slow)]
    assert np.linalg.norm(c.value - exact) <= c.err <= 0.05

    # untargeted: whatever split is returned must certify and contain
    c2, slow2 = sf.mz_search_slow(Ap, x0p, 5.0, tol=0.05)
    assert designed <= set(slow2)
    exact2 = (expm(Ap * 5.0) @ x0p)[np.array(slow2)]
    assert np.linalg.norm(c2.value - exact2) <= c2.err <= 0.05

    with pytest.raises(ValueError):                  # unreachable tolerance
        sf.mz_search_slow(Ap, x0p, 5.0, tol=1e-12)


def _rk4(f, x0, dt, n):
    x, xs = np.array(x0, float), [np.array(x0, float)]
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x.copy())
    return np.array(xs)


def test_mz_conformal_empirical():
    """Phase 4, empirical tier: nonlinear slow-fast system where no
    rigorous constants exist; the conformal bound is distribution-free
    with fail_p = 1/(n_cal+1), checked on 200 fresh draws."""
    def full(x0):        # slow x drives fast y; adiabatic reduced model
        f = lambda s: np.array([-0.2 * s[0] + 0.6 * s[1],
                                -6.0 * s[1] + s[0] - 0.3 * s[0] ** 3])
        return _rk4(f, [x0, 0.0], 0.02, 250)[:, 0]

    def red(x0):
        f = lambda s: np.array([-0.2 * s[0] + 0.1 * (s[0] - 0.3 * s[0] ** 3)])
        return _rk4(f, [x0], 0.02, 250)[:, 0]

    sampler = lambda rng: float(rng.uniform(-1, 1))
    c = sf.conformal_closure(full, red, sampler, 0.7, n_cal=99, rng=23)
    assert c.tier == sf.Tier.EMPIRICAL
    assert c.fail_p == pytest.approx(0.01)
    assert np.max(np.abs(c.value - full(0.7))) <= c.err  # typical draw held
    rng = np.random.default_rng(24)
    viol = sum(np.max(np.abs(full(x) - red(x))) > c.err
               for x in (sampler(rng) for _ in range(200)))
    assert viol <= 8            # expected <= 2 at fail_p 0.01 per draw


def _helmholtz_kernel(k):
    from scipy.special import hankel1
    def kern(t, s):
        r = np.abs(t[:, None] - s[None, :])
        return 0.25j * hankel1(0, k * r)
    return kern


def test_blackbox_hmatrix_complex_helmholtz():
    """The probe certificate extends to complex (oscillatory) kernels:
    real probes certify the real-restricted norm, x sqrt(2) covers
    complex inputs. Same every-q guarantee as the real case."""
    rng = np.random.default_rng(27)
    n = 1200
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    eps = 1e-4
    plan = sf.BlackboxHMatrix(_helmholtz_kernel(12.0), tgt, src, eps,
                              leaf_size=96, rng=rng)
    dense = _helmholtz_kernel(12.0)(tgt, src)
    for _ in range(2):
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        c, stats = plan.apply(q)
        assert np.max(np.abs(c.value - dense @ q)) \
            <= eps * np.linalg.norm(q) * (1 + 1e-9)
        assert np.linalg.norm(c.value - dense @ q) <= c.err * (1 + 1e-9)
    assert plan.fail_p > 0
    assert stats["apply_flops"] < stats["dense_flops"]


def test_helmholtz_compressibility_degrades_with_k():
    """Island cartography: the certified compression must get measurably
    worse as the kernel gets more oscillatory."""
    rng = np.random.default_rng(28)
    n = 2000
    src = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    tgt = rng.uniform(0, 1, n) + 1j * rng.uniform(0, 1, n)
    speedups = []
    for k in (2.0, 40.0):
        plan = sf.BlackboxHMatrix(_helmholtz_kernel(k), tgt, src, 1e-4,
                                  leaf_size=96, rng=rng)
        _, stats = plan.apply(np.ones(n, complex))
        speedups.append(stats["speedup"])
    assert speedups[0] > speedups[1] > 1.0


def test_demodulation_is_unitarily_vacuous():
    """Negative result, locked in so it is not re-learned: demodulating a
    kernel block by pair-direction phases is a unitary diagonal scaling,
    so its singular values — hence any SVD-based rank — are IDENTICAL.
    Per-block directional demodulation cannot extend the island; the
    genuine multi-level butterfly factorization is required."""
    rng = np.random.default_rng(32)
    k = 150.0
    src = 0.3 * (rng.uniform(-1, 1, 120) + 1j * rng.uniform(-1, 1, 120))
    tgt = 1.5 + 0.3 * (rng.uniform(-1, 1, 150) + 1j * rng.uniform(-1, 1, 150))
    K = _helmholtz_kernel(k)(tgt, src)
    u = np.mean(tgt) - np.mean(src)
    u /= abs(u)
    dT = np.exp(-1j * k * (np.conj(u) * tgt).real)
    dS = np.exp(1j * k * (np.conj(u) * src).real)
    sv = np.linalg.svd(K, compute_uv=False)
    svd = np.linalg.svd((dT[:, None] * K) * dS[None, :], compute_uv=False)
    assert np.allclose(sv, svd, rtol=1e-10)


def test_high_frequency_rank_law():
    """Island cartography: measured block rank tracks the butterfly
    parameter R = k * r_T * r_S / D — linear in k — so the H-matrix
    degrades gracefully and the butterfly regime starts near R ~ min
    block dimension (k in the thousands at this geometry)."""
    rng = np.random.default_rng(35)
    src = 0.25 * (rng.uniform(-1, 1, 200) + 1j * rng.uniform(-1, 1, 200))
    tgt = 1.2 + 0.25 * (rng.uniform(-1, 1, 200) + 1j * rng.uniform(-1, 1, 200))
    rT = float(np.max(np.abs(tgt - np.mean(tgt))))
    rS = float(np.max(np.abs(src - np.mean(src))))
    D = abs(np.mean(tgt) - np.mean(src))
    ranks = []
    for k in (30.0, 60.0, 120.0):
        sv = np.linalg.svd(_helmholtz_kernel(k)(tgt, src), compute_uv=False)
        rank = int(np.sum(sv > 1e-4 * sv[0]))
        R = k * rT * rS / D
        assert 0.5 * R < rank < 3.0 * R + 10     # rank ~ R, linear in k
        ranks.append(rank)
    assert ranks[2] > 1.5 * ranks[0]             # genuinely growing with k


_CLUSTER = {}


def _cluster_setup():
    """One 4608-point separated-cluster geometry and ONE k=1800 Hankel
    matrix, shared (via slices — sub-blocks are free) by the butterfly
    and competition tests: kernel evaluation dominates their runtime."""
    if not _CLUSTER:
        rng = np.random.default_rng(36)
        n = 4608
        src = 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
        tgt = 1.2 + 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
        _CLUSTER.update(tgt=tgt, src=src, K=_helmholtz_kernel(1800.0)(tgt, src))
    return _CLUSTER


def _lookup_kernel(K, tgt, src):
    """Wrap a precomputed matrix as a kernel callable (points -> indices)."""
    ti = {z: i for i, z in enumerate(tgt)}
    si = {z: i for i, z in enumerate(src)}
    return lambda tq, sq: K[np.ix_([ti[z] for z in tq], [si[z] for z in sq])]


def test_butterfly_certified_and_beats_plain_lowrank():
    """The multi-level butterfly: complementary low-rank with a 4-way
    (diameter-halving) ladder. At fixed R = k r_T r_S / D ~ 95 the
    advantage over plain low-rank grows with N (transfer cost is
    N-independent, plain cost is R*(m+n)): measured crossover
    1.16x -> 0.57x from N=1536 to N=4608. Certified a posteriori."""
    cl = _cluster_setup()
    rng = np.random.default_rng(40)
    ratios, r_plain = [], None
    for n in (1536, 4608):
        tgt, src, K = cl["tgt"][:n], cl["src"][:n], cl["K"][:n, :n]
        bf = sf.ButterflyBlock(K, tgt, src, levels=3, eps=1e-4,
                               rng=np.random.default_rng(1))
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        c, stats = bf.apply(q)
        exact = K @ q
        assert np.linalg.norm(c.value - exact) <= c.err * (1 + 1e-9)
        assert c.err < 0.05 * np.linalg.norm(exact)     # non-vacuous
        assert c.tier == sf.Tier.RIGOROUS and c.fail_p == pytest.approx(1e-10)
        if r_plain is None:
            # plain low-rank at MATCHED certified accuracy via the adaptive
            # certified rank finder; r_plain is N-independent at fixed R
            # (measured 94 vs 97), and reusing the n=1536 value slightly
            # UNDERSTATES plain cost at 4608 — conservative for our claim
            res = sf._compress_certified(
                K, bf.beta, 10, 10 * math.sqrt(2 / math.pi) * math.sqrt(2),
                np.random.default_rng(2))
            r_plain = res[1].shape[0]
            assert r_plain > 60                          # high-R regime
        ratios.append(stats["apply_flops"] / (r_plain * 2 * n))
        assert stats["apply_flops"] < 0.15 * n * n
    assert ratios[1] < 0.7 < ratios[0] < 1.4             # the crossover


def test_hmatrix_butterfly_competition_high_k():
    """The butterfly as a candidate in the H-matrix's per-block
    competition: two separated clusters make the root pair one big
    admissible block; at high k the butterfly must win it, beat the
    plain-only plan, and keep the every-q guarantee."""
    cl = _cluster_setup()
    rng = np.random.default_rng(38)
    n = 2304
    tgt, src = cl["tgt"][:n], cl["src"][:n]
    kern = _lookup_kernel(cl["K"][:n, :n], tgt, src)
    eps = 0.05     # loose: the probe estimator's ~500x overshoot on flat
    plan = sf.BlackboxHMatrix(kern, tgt, src, eps,      # residuals prices
                              leaf_size=4 * n, rng=np.random.default_rng(1))
    assert plan.stats["butterfly_blocks"] == 1
    plain = sf.BlackboxHMatrix(kern, tgt, src, eps, leaf_size=4 * n,
                               rng=np.random.default_rng(1),
                               try_butterfly=False)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c, stats = plan.apply(q)
    _, pstats = plain.apply(q)
    # measured 0.80 at this block size; the strong scaling claim is the
    # dedicated crossover test — here the point is the competition picks it
    assert stats["apply_flops"] < 0.85 * pstats["apply_flops"]
    dense = cl["K"][:n, :n]
    assert np.max(np.abs(c.value - dense @ q)) \
        <= eps * np.linalg.norm(q) * (1 + 1e-9)
    assert np.linalg.norm(c.value - dense @ q) <= c.err * (1 + 1e-9)


def test_hmatrix_butterfly_competition_rejects_at_low_k():
    """Same geometry, low k: the butterfly is tried and correctly loses
    the competition to plain low-rank."""
    rng = np.random.default_rng(39)
    n = 1024
    src = 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
    tgt = 1.2 + 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
    kern = _helmholtz_kernel(30.0)
    plan = sf.BlackboxHMatrix(kern, tgt, src, 0.05, leaf_size=4 * n,
                              rng=np.random.default_rng(1))
    assert plan.stats["butterfly_blocks"] == 0
    assert plan.stats["lr_blocks"] == 1
    dense = kern(tgt, src)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c, _ = plan.apply(q)
    assert np.max(np.abs(c.value - dense @ q)) \
        <= 0.05 * np.linalg.norm(q) * (1 + 1e-9)


def test_butterfly_low_R_is_no_worse_than_graceful():
    """Sanity at low R (Laplace-like regime): the butterfly still
    certifies; it just has no structural advantage to exploit."""
    rng = np.random.default_rng(37)
    n = 512
    src = 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
    tgt = 1.2 + 0.25 * (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n))
    K = _helmholtz_kernel(30.0)(tgt, src)
    bf = sf.ButterflyBlock(K, tgt, src, levels=2, eps=1e-4, rng=rng)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c, _ = bf.apply(q)
    assert np.linalg.norm(c.value - K @ q) <= c.err * (1 + 1e-9)


def test_helmholtz_scatter_certified():
    """CEM beachhead: certified far field of a weak penetrable scatterer.
    Truth is the dense solve of the same discrete system (the declared
    scope; continuum discretization error is the named gap)."""
    contrast = lambda x, y: 0.12 * np.exp(-(x**2 + y**2) / 0.25)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    c, stats = sf.helmholtz_scatter_farfield(6.0, contrast, 24, 2.0,
                                             angles, tol=1e-8)
    pts, m, K = sf._helmholtz_K(6.0, contrast, 24, 2.0)
    ui = np.exp(1j * 6.0 * pts.real)
    u_true = np.linalg.solve(np.eye(len(pts)) - K, ui)
    gamma = np.exp(1j * np.pi / 4) / math.sqrt(8 * math.pi * 6.0)
    h2 = (2.0 / 24) ** 2
    for i, th in enumerate(angles):
        w = gamma * 36.0 * m * h2 * np.exp(-1j * 6.0 * (
            math.cos(th) * pts.real + math.sin(th) * pts.imag))
        assert abs(c.value[i] - w @ u_true) <= stats["angle_bound"][i]
    assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0  # Schur is deterministic
    # the probe bound really dominates the operator norm
    assert stats["beta"] >= np.linalg.norm(K, 2)
    # solver depth scales with the question's precision
    _, s1 = sf.helmholtz_scatter_farfield(6.0, contrast, 24, 2.0, angles,
                                          tol=1e-4)
    assert s1["n_terms"] < stats["n_terms"]


def test_helmholtz_scatter_refuses_strong_contrast():
    """Outside the certified weak-scattering region (||K|| >= 1): refuse,
    do not extrapolate."""
    contrast = lambda x, y: 3.0 * np.exp(-(x**2 + y**2) / 0.25)
    with pytest.raises(ValueError):
        sf.helmholtz_scatter_farfield(6.0, contrast, 24, 2.0,
                                      np.array([0.0]), tol=1e-6)


def _heisenberg_chain(N):
    """Spin-1/2 Heisenberg open chain: H = sum_i S_i . S_{i+1}. Real
    symmetric in the computational basis (the two i's in Sy x Sy cancel)."""
    half = [np.array([[0, 1], [1, 0]]) / 2,
            np.array([[0, -1j], [1j, 0]]) / 2,
            np.array([[1, 0], [0, -1]]) / 2]
    H = np.zeros((2 ** N, 2 ** N), complex)
    for i in range(N - 1):
        for s in half:
            op = np.eye(1)
            for j in range(N):
                op = np.kron(op, s if j in (i, i + 1) else np.eye(2))
            H += op
    assert np.max(np.abs(H.imag)) < 1e-14
    return H.real


def test_eigen_bracket_heisenberg():
    """The chemistry energy bracket, matrix tier: certified two-sided
    interval on the ground energy of a real many-body Hamiltonian,
    verified against full diagonalization."""
    H = _heisenberg_chain(10)
    c = sf.eigen_bracket(H)
    truth = float(np.linalg.eigvalsh(H)[0])
    assert c.value - c.err <= truth <= c.value + c.err
    assert c.err < 1e-6                       # far below chemical accuracy
    assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
    assert "cholesky" in c.provenance[0]
    # physics sanity: per-bond energy near the Bethe value 1/4 - ln 2
    assert -0.50 < truth / 9 < -0.35


def test_eigen_bracket_complex_and_tolerance():
    rng = np.random.default_rng(41)
    A = rng.standard_normal((300, 300)) + 1j * rng.standard_normal((300, 300))
    H = (A + A.conj().T) / 2
    c = sf.eigen_bracket(H)
    truth = float(np.linalg.eigvalsh(H)[0])
    assert c.value - c.err <= truth <= c.value + c.err
    assert c.err < 1e-6 * abs(truth)
    with pytest.raises(ValueError):           # below the FP floor: refuse
        sf.eigen_bracket(H, tol=1e-18)


def test_chain_bracket_matches_exact_when_formable():
    """The block-marginal bracket cross-validated against exact
    diagonalization where the chain is still formable."""
    truth = float(np.linalg.eigvalsh(_heisenberg_chain(10))[0])
    c = sf.heisenberg_chain_bracket(10, ell=8)
    assert c.value - c.err <= truth <= c.value + c.err
    assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
    assert 2 * c.err < 0.12 * abs(truth)     # a real bracket, not vacuous


def test_chain_bracket_past_formable():
    """The point of the marginal relaxation: a certified two-sided bracket
    on the ground energy of a 2^200-dimensional Hamiltonian, at cost
    2^ell independent of N. The thermodynamic Bethe value must fall
    inside the per-bond bracket."""
    c = sf.heisenberg_chain_bracket(200, ell=8)
    per_bond_lo = (c.value - c.err) / 199
    per_bond_hi = (c.value + c.err) / 199
    bethe = 0.25 - math.log(2)               # -0.443147...
    assert per_bond_lo <= bethe <= per_bond_hi
    assert per_bond_hi - per_bond_lo < 0.08  # ~13% relaxation gap at ell=8
    assert "marginal" in c.provenance[0]


def test_correction_multipliers_tighten_lower_bound():
    """The SDP dual, ascended over the telescoping-correction family:
    optimized multipliers must tighten the certified lower bound, and the
    bracket must stay valid (Bethe density inside)."""
    plain = sf.heisenberg_chain_bracket(200, ell=6, correction_iters=0)
    corr = sf.heisenberg_chain_bracket(200, ell=6)
    gain = (corr.value - corr.err) - (plain.value - plain.err)
    assert gain > 4.0    # measured ~6.5 at ell=6, N=200 (~58% of gap closed)
    bethe = 0.25 - math.log(2)
    assert (corr.value - corr.err) / 199 <= bethe \
        <= (corr.value + corr.err) / 199


def _interior_lambda(ell, C):
    Hw = sf._heis_window((1.0 / (ell - 1),) * (ell - 1))
    I2 = np.eye(2)
    return float(np.linalg.eigvalsh(Hw + np.kron(C, I2)
                                    - np.kron(I2, C))[0])


def test_dual_exhausted_at_fixed_ell():
    """Locked-in findings: (a) the bundle converges — doubling oracle
    calls moves the dual value < 1e-3; (b) decomposition-weight freedom
    is ABSORBED by the correction family — joint (C, weights) ascent
    cannot beat the single-C bundle optimum. With translation invariance
    WLOG in the bulk and non-consecutive overlaps redundant, the
    single-overlap family exhausts the fully general dual at fixed ell;
    the residual gap is the relaxation level, priced by ell."""
    l80 = _interior_lambda(6, sf._chain_correction(6, 80))
    l160 = _interior_lambda(6, sf._chain_correction(6, 160))
    assert abs(l160 - l80) < 1e-3
    # joint (C, weights) supergradient ascent at ell=4
    ell, nb = 4, 3
    d, I2 = 2 ** (ell - 1), np.eye(2)
    hj = [sf._heis_window(tuple(1.0 if k == j else 0.0 for k in range(nb)))
          for j in range(nb)]
    a, C = np.full(nb, 1.0 / nb), np.zeros((d, d))
    joint = -np.inf
    for k in range(600):
        M = sum(a[j] * hj[j] for j in range(nb))             + np.kron(C, I2) - np.kron(I2, C)
        lam, V = np.linalg.eigh(M)
        joint = max(joint, lam[0])
        v = V[:, 0]
        Vl, Vr = v.reshape(d, 2), v.reshape(2, d)
        ga = np.array([v @ (hj[j] @ v) for j in range(nb)])
        C = C + 0.4 / math.sqrt(k + 1) * (Vl @ Vl.T - Vr.T @ Vr)
        a = a + 0.05 / math.sqrt(k + 1) * (ga - ga.mean())
    bundle = _interior_lambda(4, sf._chain_correction(4, 80))
    assert bundle >= joint - 1e-3


def test_chain_bracket_tightens_with_ell():
    """Cost scales with the precision of the question: a longer window
    (2^ell diagonalizations) buys a tighter certified bracket."""
    wide = sf.heisenberg_chain_bracket(200, ell=4)
    tight = sf.heisenberg_chain_bracket(200, ell=8)
    assert tight.err < 0.75 * wide.err


def test_h2_integrals_against_quadrature():
    """The closed-form s-Gaussian integrals against direct 3D grid
    quadrature — catches every convention and prefactor bug."""
    R = 1.4
    S, h, eri, _ = sf._h2_integrals(R)
    g = np.arange(-7.03, 8.0, 0.14)      # nuclei strictly off grid points
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    dv = 0.14 ** 3
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    def ao(center):
        r2 = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
        out = np.zeros_like(X)
        for a, cn in sf._sto3g_h():
            out += cn * np.exp(-a * r2)
        return out

    def grad_ao(center):
        r2 = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
        gx = gy = gz = 0.0
        for a, cn in sf._sto3g_h():
            e = cn * np.exp(-a * r2)
            gx = gx - 2 * a * (X - center[0]) * e
            gy = gy - 2 * a * (Y - center[1]) * e
            gz = gz - 2 * a * (Z - center[2]) * e
        return gx, gy, gz

    phiA, phiB = ao(A), ao(B)
    assert np.sum(phiA * phiA) * dv == pytest.approx(S[0, 0], abs=1e-5)
    assert np.sum(phiA * phiB) * dv == pytest.approx(S[0, 1], abs=1e-5)
    # core h = kinetic (independent gradient quadrature) + nuclear
    # attraction (singularity integrable, nuclei off grid points)
    gA, gB = grad_ao(A), grad_ao(B)
    Tnum = 0.5 * np.sum(gA[0] * gB[0] + gA[1] * gB[1] + gA[2] * gB[2]) * dv
    rA = np.sqrt(X**2 + Y**2 + Z**2)
    rB = np.sqrt(X**2 + Y**2 + (Z - R)**2)
    Vnum = -np.sum(phiA * phiB * (1 / rA + 1 / rB)) * dv
    assert Tnum + Vnum == pytest.approx(h[0, 1], abs=3e-3)
    # ERI (00|11) via the closed-form erf potential of the B-B pair density
    from scipy.special import erf
    VB = np.zeros_like(X)
    s = np.sqrt((X - B[0])**2 + (Y - B[1])**2 + (Z - B[2])**2)
    for a, ca in sf._sto3g_h():
        for b, cb in sf._sto3g_h():
            q = a + b
            VB += ca * cb * (np.pi / q) ** 1.5 * np.where(
                s > 1e-8, erf(np.sqrt(q) * s) / np.maximum(s, 1e-30),
                2 * math.sqrt(q / math.pi))
    eri0011 = np.sum(phiA * phiA * VB) * dv
    assert eri0011 == pytest.approx(eri[0, 0, 1, 1], abs=2e-4)


def test_h2_bracket_dissociation_and_equilibrium():
    """End-to-end: the certified bracket must reproduce two internal
    truths — dissociation to two isolated atoms, and the independent
    2x2 MO-basis CI (exact FCI for H2/STO-3G by parity)."""
    # at R=50 monopole terms cancel exactly for s-orbitals, so the total
    # must equal two isolated atoms to exponential accuracy
    c_far = sf.h2_energy_bracket(50.0)
    assert abs(c_far.value - 2 * sf.hydrogen_atom_energy()) < 1e-6
    c_eq = sf.h2_energy_bracket(1.4)
    assert c_eq.value + c_eq.err < c_far.value - c_far.err   # binding
    # independent truth: MO-basis 2x2 CI from the same integrals
    S, h, eri, enuc = sf._h2_integrals(1.4)
    # NOTE: published STO-3G contractions normalize the AO only to ~1e-8
    # (S11 != 1 exactly), so the MO normalization must use S11, not 1 —
    # getting this wrong shifts the CI by 7e-9 and the bracket catches it
    Cg = np.array([1, 1]) / math.sqrt(2 * (S[0, 0] + S[0, 1]))
    Cu = np.array([1, -1]) / math.sqrt(2 * (S[0, 0] - S[0, 1]))
    hg, hu = Cg @ h @ Cg, Cu @ h @ Cu
    mo = lambda P, Q, R_, S_: np.einsum("pqrs,p,q,r,s->", eri, P, Q, R_, S_)
    H2x2 = np.array([[2 * hg + mo(Cg, Cg, Cg, Cg), mo(Cg, Cu, Cg, Cu)],
                     [mo(Cg, Cu, Cg, Cu), 2 * hu + mo(Cu, Cu, Cu, Cu)]])
    e_ci = float(np.linalg.eigvalsh(H2x2)[0]) + enuc
    assert abs(c_eq.value - e_ci) <= c_eq.err + 1e-9
    assert c_eq.err < 1e-8
    assert c_eq.tier == sf.Tier.RIGOROUS
    # loose cross-reference to the published FCI/STO-3G value ~ -1.1373 Ha
    assert -1.141 < c_eq.value < -1.133


def test_md_reduces_to_s_closed_forms():
    """At l=0 the McMurchie-Davidson engine must reproduce the closed-form
    s-integrals to machine precision — last turn's code is the oracle."""
    R = 1.4
    S0, h0, eri0, _ = sf._h2_integrals(R)
    prims = sf._STO3G_H_RAW           # raw: _md_integrals applies norms
    shells = [((0.0, 0.0, 0.0), (0, 0, 0), prims),
              ((0.0, 0.0, R), (0, 0, 0), prims)]
    atoms = [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, R))]
    S, h, eri, enuc = sf._md_integrals(atoms, shells)
    assert np.max(np.abs(S - S0)) < 1e-12
    assert np.max(np.abs(h - h0)) < 1e-12
    assert np.max(np.abs(eri - eri0)) < 1e-12
    assert enuc == pytest.approx(1.0 / R)


def test_md_p_integrals_against_quadrature():
    """Genuine p-orbital integrals against 3D grid quadrature."""
    R = 1.4
    a_p = 1.1                                # standard H polarization exp
    pz = ((0.0, 0.0, 0.0), (0, 0, 1), ((a_p, 1.0),))
    pzB = ((0.0, 0.0, R), (0, 0, 1), ((a_p, 1.0),))
    s_ = ((0.0, 0.0, 0.0), (0, 0, 0), sf._STO3G_H_RAW)
    atoms = [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, R))]
    S, h, eri, _ = sf._md_integrals(atoms, [s_, pz, pzB])
    g = np.arange(-7.03, 8.0, 0.14)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    dv = 0.14 ** 3
    norm_p = (2 * a_p / math.pi) ** 0.75 * 2 * math.sqrt(a_p)
    pzA_g = norm_p * Z * np.exp(-a_p * (X**2 + Y**2 + Z**2))
    pzB_g = norm_p * (Z - R) * np.exp(-a_p * (X**2 + Y**2 + (Z - R)**2))
    sA_g = np.zeros_like(X)
    for a, cn in sf._sto3g_h():
        sA_g += cn * np.exp(-a * (X**2 + Y**2 + Z**2))
    assert np.sum(pzA_g * pzA_g) * dv == pytest.approx(S[1, 1], abs=1e-5)
    assert np.sum(pzA_g * pzB_g) * dv == pytest.approx(S[1, 2], abs=1e-5)
    assert np.sum(sA_g * pzB_g) * dv == pytest.approx(S[0, 2], abs=1e-5)
    # s-pz nuclear + kinetic jointly through the core Hamiltonian
    gz_pzB = norm_p * (1 - 2 * a_p * (Z - R)**2) \
        * np.exp(-a_p * (X**2 + Y**2 + (Z - R)**2))
    gx_pzB = norm_p * (Z - R) * (-2 * a_p * X) \
        * np.exp(-a_p * (X**2 + Y**2 + (Z - R)**2))
    gy_pzB = norm_p * (Z - R) * (-2 * a_p * Y) \
        * np.exp(-a_p * (X**2 + Y**2 + (Z - R)**2))
    gx_s = gy_s = gz_s = None
    gs = []
    for a, cn in sf._sto3g_h():
        gs.append((a, cn))
    gx_s = sum(-2 * a * X * cn * np.exp(-a * (X**2 + Y**2 + Z**2))
               for a, cn in gs)
    gy_s = sum(-2 * a * Y * cn * np.exp(-a * (X**2 + Y**2 + Z**2))
               for a, cn in gs)
    gz_s = sum(-2 * a * Z * cn * np.exp(-a * (X**2 + Y**2 + Z**2))
               for a, cn in gs)
    Tnum = 0.5 * np.sum(gx_s * gx_pzB + gy_s * gy_pzB + gz_s * gz_pzB) * dv
    rA = np.sqrt(X**2 + Y**2 + Z**2)
    rB = np.sqrt(X**2 + Y**2 + (Z - R)**2)
    Vnum = -np.sum(sA_g * pzB_g * (1 / rA + 1 / rB)) * dv
    assert Tnum + Vnum == pytest.approx(h[0, 2], abs=3e-3)
    # one p-involving ERI via the erf potential of the s-s pair density
    from scipy.special import erf
    VA = np.zeros_like(X)
    s = np.sqrt(X**2 + Y**2 + Z**2)
    for a, ca in sf._sto3g_h():
        for b, cb in sf._sto3g_h():
            q = a + b
            VA += ca * cb * (np.pi / q) ** 1.5 * np.where(
                s > 1e-8, erf(np.sqrt(q) * s) / np.maximum(s, 1e-30),
                2 * math.sqrt(q / math.pi))
    assert np.sum(pzA_g * pzB_g * VA) * dv \
        == pytest.approx(eri[1, 2, 0, 0], abs=2e-4)


def test_h2_polarized_bracket_strictly_below_s_only():
    """The certificates prove the physics: adding pz polarization must
    lower the FCI energy, and with 1e-13 bracket widths 'strictly below'
    is a theorem about the two intervals."""
    R = 1.4
    c_s = sf.h2_energy_bracket(R)
    c_sp = sf.h2_polarized_bracket(R)
    assert c_sp.value + c_sp.err < c_s.value - c_s.err
    lowering = c_s.value - c_sp.value
    assert 1e-4 < lowering < 0.05          # mHa-scale, physically sane
    assert c_sp.tier == sf.Tier.RIGOROUS


def test_h_chain_bracket_vs_exact_when_formable():
    """The marriage, gated by exact FCI: for H4 the full 256-dim Fock
    Hamiltonian is formable, and the window-decomposed bracket must
    contain its exact ground energy."""
    truth = float(np.linalg.eigvalsh(sf.h_chain_fock_hamiltonian(4, 1.8))[0])
    c = sf.h_chain_bracket(4, 1.8, ell=3)
    assert c.value - c.err <= truth <= c.value + c.err
    assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
    assert 2 * c.err < 0.3 * abs(truth)      # 143 mHa/atom with CS absorption


def test_h_chain_bracket_past_formable():
    """H10: a 2^20-dimensional molecular Fock space, bracketed at window
    cost. No dense truth exists; the checks are internal consistency and
    physical sanity of the certified per-atom energy."""
    c = sf.h_chain_bracket(10, 1.8, ell=3)
    assert c.err > 0 and c.value == c.value  # well-formed
    per_atom_lo = (c.value - c.err) / 10
    per_atom_hi = (c.value + c.err) / 10
    assert per_atom_lo < -0.45 and per_atom_hi > -0.65
    assert per_atom_hi < -0.30               # upper genuinely binds
    assert per_atom_hi - per_atom_lo < 0.55  # CS absorption: was 0.636
    assert "marginal" in c.provenance[0]


def test_molecular_multipliers_tighten():
    """The bundle multipliers, ported to molecular windows (shared C on
    atom overlaps, valid for ANY Hermitian C by qubit-level telescoping):
    the certified lower bound must rise, containment must survive."""
    from scipy.sparse.linalg import eigsh
    truth = float(eigsh(sf.h_chain_fock_hamiltonian(6, 1.8),
                        k=1, which="SA")[0][0])
    plain = sf.h_chain_bracket(6, 1.8, ell=3, correction_iters=0)
    corr = sf.h_chain_bracket(6, 1.8, ell=3)
    assert corr.value - corr.err <= truth <= corr.value + corr.err
    gain = (corr.value - corr.err) - (plain.value - plain.err)
    # per-overlap multipliers: 0.206 Ha measured at H6/ell=3, 3x the
    # shared-C version (the finite chain's edge effects live in the
    # per-overlap freedom); on top of CS absorption
    assert gain > 0.15


def test_eps_loop_never_hurts():
    """Outer eps loop for the CS weights, locked-in findings: the naive
    balance update eps* = sqrt(<R>/<L>) DEGRADES the bound (measured
    -5 mHa/atom at H6 — it optimizes against the current window
    minimizers; lambda_min then re-minimizes elsewhere, so it is not an
    ascent step), and with damped updates plus best-by-measured-total
    tracking the loop is monotone-safe — the eps=1 start is in the
    candidate set, so it can never end worse. Measured optimum on this
    system IS eps=1: the per-term balancing family has nothing to give
    here; the guarantee is the deliverable."""
    base = sf.h_chain_bracket(6, 1.8, ell=3, cs_rounds=0)
    bal = sf.h_chain_bracket(6, 1.8, ell=3, cs_rounds=4)
    assert (bal.value - bal.err) >= (base.value - base.err) - 1e-9


def test_h_chain_bracket_tightens_with_ell():
    wide = sf.h_chain_bracket(8, 1.8, ell=2)
    tight = sf.h_chain_bracket(8, 1.8, ell=3)
    assert tight.err < wide.err


def test_fermion_assemble_matches_jw_products():
    """The bit-arithmetic assembler (each ladder string is a signed
    partial permutation — no matrix products) against explicit JW
    matrix products, including string-parity signs."""
    rng = np.random.default_rng(42)
    nq = 6
    ann = sf._jw_ann(nq)
    terms = []
    ref = np.zeros((2 ** nq, 2 ** nq))
    for _ in range(30):
        k = rng.integers(2, 5)
        modes = rng.choice(nq, size=k, replace=False)
        dags = rng.integers(0, 2, size=k).astype(bool)
        coef = float(rng.standard_normal())
        terms.append((coef, [(int(m), bool(d))
                             for m, d in zip(modes, dags)]))
        M = np.eye(2 ** nq)
        for m, d in reversed(list(zip(modes, dags))):
            op = ann[int(m)].toarray()
            M = (op.T if d else op) @ M
        ref += coef * M
    got = sf._fermion_assemble(nq, terms).toarray()
    assert np.max(np.abs(got - ref)) < 1e-12


def test_sectored_certification_matches_dense():
    """Occupation-sector certification (the ell=7 enabler — every window
    term conserves N_up/N_down, so the 4^ell space block-diagonalizes;
    dense Cholesky runs per sector) must reproduce the dense
    eigen_bracket on the same operator, with a rigorous ~0 off-sector
    penalty rather than an assumption."""
    from scipy import sparse
    T, V, eri, _ = sf._h_chain_basis(6, 1.8)
    idx = np.ix_(range(3), range(3))
    hw = (T + V.sum(0))[idx]
    Hw = sf._window_operator(hw, eri[np.ix_(range(3), range(3),
                                            range(3), range(3))],
                             np.zeros(3), np.full(3, 0.1), 0.3)
    lo, up = sf._eigen_bracket_sectored(Hw.tocsr())
    dense = sf.eigen_bracket(np.asarray(Hw.todense()))
    assert abs(lo - (dense.value - dense.err)) < 1e-9
    assert abs(up - (dense.value + dense.err)) < 1e-9
    assert lo <= dense.value <= up


def test_fp32_certification_prep():
    """GPU prep: fp32 Cholesky certification with honestly widened pads
    — the MEASURED casting error ||H - fl32(H)||_F plus Higham margins
    at fp32 eps. Contained, width at the predicted ~1e-3 pad scale,
    negligible against mHa brackets. Consumer-GPU fp64 is ~1:64
    throttled; fp32-with-pads is what makes the GPU pay."""
    H = _heisenberg_chain(10)
    truth = float(np.linalg.eigvalsh(H)[0])
    c = sf.eigen_bracket(H, fp32=True)
    assert c.value - c.err <= truth <= c.value + c.err
    assert 1e-4 < c.err < 2e-2               # pads present, not vacuous
    from scipy.sparse.linalg import eigsh
    t6 = float(eigsh(sf.h_chain_fock_hamiltonian(6, 1.8),
                     k=1, which="SA")[0][0])
    cb = sf.h_chain_bracket(6, 1.8, ell=4, fp32=True)
    assert cb.value - cb.err <= t6 <= cb.value + cb.err
    c64 = sf.h_chain_bracket(6, 1.8, ell=4)
    assert cb.err < c64.err + 0.01           # inflated by pads only
    with pytest.raises(ImportError):         # no cupy on this box
        sf.use_gpu(True)


def test_h_chain_ell5_hierarchy_knob():
    """The hierarchy knob at ell=5 (1024-dim windows): steep payoff —
    H6 width/atom 244 (ell=3) -> 89 (ell=4) -> 66 mHa (ell=5), still
    containing the exact FCI energy."""
    from scipy.sparse.linalg import eigsh
    truth = float(eigsh(sf.h_chain_fock_hamiltonian(6, 1.8),
                        k=1, which="SA")[0][0])
    c5 = sf.h_chain_bracket(6, 1.8, ell=5)
    c4 = sf.h_chain_bracket(6, 1.8, ell=4)
    assert c5.value - c5.err <= truth <= c5.value + c5.err
    assert c5.err < 0.85 * c4.err
    assert 2 * c5.err / 6 < 0.075            # <= 75 mHa/atom


def test_reduced_basis_bracket_certified_across_sweep():
    """Certified reduced basis / eigenvector continuation: 6 offline
    snapshots of the transverse-field Ising chain, then a 100-point
    sweep across the quantum phase transition at k x k online cost —
    every point's bracket must contain the exact ground energy."""
    H0, H1 = sf.tfi_chain(10)
    thetas = np.linspace(0.0, 2.0, 6)
    sur = sf.reduced_basis_surrogate(H0, H1, thetas)
    widths = []
    for g in np.linspace(0.0, 2.0, 100):
        c = sf.reduced_basis_bracket(sur, g)
        truth = float(np.linalg.eigvalsh(H0 + g * H1)[0])
        assert c.value - c.err <= truth <= c.value + c.err
        assert c.tier == sf.Tier.RIGOROUS
        widths.append(2 * c.err)
    assert max(widths) < 0.6          # non-vacuous everywhere (N=10 scale)
    # the EC magic: the variational UPPER from 6 snapshots is excellent
    g = 1.0                            # the critical point
    c = sf.reduced_basis_bracket(sur, g)
    truth = float(np.linalg.eigvalsh(H0 + g * H1)[0])
    assert (c.value + c.err) - truth < 5e-3


def test_reduced_basis_refuses_extrapolation():
    """Concavity certifies chords INSIDE the snapshot hull only:
    extrapolation is refused, not guessed."""
    H0, H1 = sf.tfi_chain(6)
    sur = sf.reduced_basis_surrogate(H0, H1, np.linspace(0.5, 1.5, 4))
    with pytest.raises(ValueError):
        sf.reduced_basis_bracket(sur, 2.0)
    with pytest.raises(ValueError):
        sf.reduced_basis_bracket(sur, 0.1)


def test_reduced_basis_tightens_with_snapshots():
    """The knob: more offline snapshots buy a tighter certified band."""
    H0, H1 = sf.tfi_chain(8)
    def mean_width(k):
        sur = sf.reduced_basis_surrogate(H0, H1, np.linspace(0.0, 2.0, k))
        return float(np.mean([2 * sf.reduced_basis_bracket(sur, g).err
                              for g in np.linspace(0.0, 2.0, 50)]))
    assert mean_width(9) < 0.5 * mean_width(3)


def test_end_to_end_chain():
    """The Phase 0 deliverable: a 3-rewrite chain (compress, project, truncate)
    whose composed bound contains the true end-to-end error."""
    rng = np.random.default_rng(3)
    # near sources seen through a kernel matrix, far sources via multipole
    tx = rng.uniform(0, 1, 80)                     # targets on [0,1]
    near_y = rng.uniform(2, 3, 60)                 # near cluster
    far_src = 10.0 + rng.uniform(-0.5, 0.5, 50) + 1j * rng.uniform(-0.5, 0.5, 50)
    qn, qf = rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 50)
    K = np.log(np.abs(tx[:, None] - near_y[None, :]))

    # rewrite 1: compress the near-field matvec to rank 8
    near = sf.lowrank_matvec(K, qn, 8)
    # rewrite 2: project onto the observable mean(phi); 1/sqrt(n)-Lipschitz in 2-norm
    obs = sf.lipschitz(np.mean, 1.0 / math.sqrt(len(tx)), near, "mean")
    # rewrite 3: far-field at probe point via truncated multipole, order 6
    probe = 0.5 + 0.0j
    far = sf.multipole_far_potential(qf, far_src, 10.0 + 0.0j, probe, 6)
    answer = obs + far

    truth = float(np.mean(K @ qn)) + float(np.sum(qf * np.log(np.abs(probe - far_src))))
    actual_err = abs(answer.value - truth)
    assert actual_err <= answer.err * (1 + 1e-9) + 1e-12
    assert answer.err < 1e-2  # and the bound is tight enough to be useful
    assert answer.tier == sf.Tier.RIGOROUS
    assert "add" in answer.provenance and "mean" in answer.provenance


def _tfi_full_zt(n, site, t, g):
    """Exact <Z_site(t)> from the all-up state of the full 2^n chain,
    built independently of the cone code (same embed convention as
    tfi_chain: site 0 most significant)."""
    H0, H1 = sf.tfi_chain(n)
    lam, V = np.linalg.eigh(H0 + g * H1)
    psi0 = np.zeros(2 ** n)
    psi0[0] = 1.0
    psi_t = V @ (np.exp(-1j * lam * t) * (V.T @ psi0))
    z = 1.0 - 2.0 * ((np.arange(2 ** n) >> (n - 1 - site)) & 1)
    return float(np.real(np.vdot(psi_t, z * psi_t)))


def test_lr_dispatch_certified_vs_exact():
    """Quantum-dynamics dispatch: the cone answer with its a-posteriori
    boundary-commutator certificate must contain the exact full-chain
    value — mid-chain and with the cone clipped at the chain end, at
    the critical point g=1 where spreading is fastest."""
    n, g = 10, 1.0
    for site in (4, 1):
        for t in (0.4, 0.9):
            truth = _tfi_full_zt(n, site, t, g)
            c = sf.tfi_quench_dispatch(n, site, t, tol=2e-2, g=g,
                                       n_steps=300)
            assert abs(c.value - truth) <= c.err
            assert c.err <= 2e-2
            assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
            assert "lr-cone" in c.provenance[0]


def test_lr_cone_tightens_with_radius():
    """A wider cone must certify tighter (the light cone has farther to
    travel), and every radius must still contain the truth."""
    n, site, t, g = 10, 4, 0.9, 1.0
    truth = _tfi_full_zt(n, site, t, g)
    v2, e2 = sf._lr_cone_run(n, site, t, 1.0, g, 2, 200)
    v3, e3 = sf._lr_cone_run(n, site, t, 1.0, g, 3, 200)
    assert abs(v2 - truth) <= e2 and abs(v3 - truth) <= e3
    assert 0.0 < e3 < e2


def test_lr_dispatch_refuses_beyond_budget():
    """t large enough that the light cone outruns every affordable cone:
    dispatch must refuse with the measured ladder, not extrapolate."""
    with pytest.raises(ValueError, match="lr-dispatch"):
        sf.tfi_quench_dispatch(50, 25, 3.0, tol=1e-6, max_dim=256,
                               n_steps=200)


def test_lr_dispatch_n_independent():
    """The whole point: the certified answer must not depend on the
    chain length — a 10^6-site chain costs the same as 2001 sites and
    returns the bit-identical certificate."""
    c1 = sf.tfi_quench_dispatch(2001, 1000, 0.6, tol=1e-3, n_steps=300)
    c2 = sf.tfi_quench_dispatch(10 ** 6, 500_000, 0.6, tol=1e-3,
                                n_steps=300)
    assert c1.value == c2.value and c1.err == c2.err


def test_lr_quadrature_honesty():
    """Coarser time quadrature must cost certified error (the measured
    derivative-ladder pad grows with the step), never gain it."""
    _, e_coarse = sf._lr_cone_run(30, 15, 0.8, 1.0, 1.0, 3, 60)
    _, e_fine = sf._lr_cone_run(30, 15, 0.8, 1.0, 1.0, 3, 480)
    assert e_fine < e_coarse


def test_asymptotic_tier_ships():
    """The first ASYMPTOTIC certificate: order-1 guiding-center drift,
    exponent proven, constant measured on a cheap large-eps ladder,
    extrapolated to a small query eps — and verified here against an
    expensive full kinetic solve the certificate never used."""
    eps = 0.02
    c = sf.gc_drift_asymptotic(eps, order=1)
    truth = sf._gc_truth_cached(eps, 0.3, 1.0, 25.0)
    assert c.tier == sf.Tier.ASYMPTOTIC and c.fail_p == 0.0
    assert abs(c.value - truth) <= c.err
    assert c.err < 0.1 * abs(truth)          # useful, not just valid
    assert "exponent 2 proven" in c.provenance[0]
    assert "measured" in c.provenance[0]


def test_gc_proven_exponent_fingerprint():
    """The theorem's fingerprint in the measurements. Order 0: the
    error constant E/eps is flat across a factor-4 range (secular
    drift, clean slope 1). Order 1: the coefficient is gyrophase-
    oscillatory, so the fingerprint is the ENVELOPE — E/eps^2 stays
    bounded and does not grow toward small eps."""
    import math as m
    eps_ladder = (0.16, 0.08, 0.04, 0.02)
    E0 = [abs(sf._gc_truth_cached(e, 0.3, 1.0, 25.0)) for e in eps_ladder]
    slope0 = m.log(E0[0] / E0[-1]) / m.log(eps_ladder[0] / eps_ladder[-1])
    assert 0.9 < slope0 < 1.1, slope0
    chat = [abs(sf._gc_prediction(1, e, 0.3, 1.0, 25.0)
                - sf._gc_truth_cached(e, 0.3, 1.0, 25.0)) / e ** 2
            for e in eps_ladder]
    assert max(chat) < 0.1                       # bounded envelope
    assert chat[-1] <= 2.0 * max(chat[:-1])      # no growth at the floor


def test_asymptotic_refuses():
    """No extrapolation above the calibration ladder; and the generic
    certifier must refuse a claimed exponent the data contradict —
    here truth ~ eps fed to a claimed k=2, so the measured constant
    grows 2x per rung toward the floor."""
    with pytest.raises(ValueError, match="ladder"):
        sf.gc_drift_asymptotic(0.3, order=1, ladder=(0.16, 0.08, 0.04))
    with pytest.raises(ValueError, match="grows"):
        sf.asymptotic_extrapolate(lambda e: 0.0, lambda e: e, 0.05, 2,
                                  (0.8, 0.4, 0.2, 0.1))


def test_plasma_hierarchy_dispatch():
    """Certified dispatch along the reduction hierarchy: a loose tol is
    served by order 0 (free), a tighter one escalates to order 1, and
    an impossible one refuses with the priced full-kinetic fallback."""
    c0 = sf.gc_drift_dispatch(0.004, tol=0.05)
    assert "order=0" in c0.provenance[0]
    c1 = sf.gc_drift_dispatch(0.02, tol=2e-3)
    assert "order=1" in c1.provenance[0] and c1.err <= 2e-3
    with pytest.raises(ValueError, match="kinetic"):
        sf.gc_drift_dispatch(0.02, tol=1e-9)


def test_asymptotic_tier_composes_downward():
    """IR check: ASYMPTOTIC composed with RIGOROUS degrades to
    ASYMPTOTIC — the weakest link names the chain."""
    c = sf.gc_drift_asymptotic(0.02, order=1)
    rig = sf.Certified(1.0, 0.0, sf.Tier.RIGOROUS, ("const",))
    assert (c + rig).tier == sf.Tier.ASYMPTOTIC


def test_sos_exact_checker():
    """The exact-rational SOS checker: accepts a hand-built SOS with a
    known Gram, refuses an indefinite polynomial. No floats anywhere
    in the verdict."""
    from fractions import Fraction as F
    # (x - 2y)^2 + (xy - 1)^2 = x^2 - 6xy + 4y^2 + x^2y^2 + 1  >= 0
    S = {(2, 0, 0): F(1), (1, 1, 0): F(-6), (0, 2, 0): F(4),
         (2, 2, 0): F(1), (0, 0, 0): F(1)}
    assert sf._sos_exact_check(S) is True
    bad = {(2, 0, 0): F(1), (1, 1, 0): F(-3), (0, 2, 0): F(1)}
    assert sf._sos_exact_check(bad) is False


def test_lorenz_z_quadratic_exact():
    """Degree-2 rung, fully by hand: the SOS proof of <z> <= rho on
    every trajectory, certified in exact rational arithmetic, and the
    fixed-point witness <z> = rho - 1 closing the bracket to width 1/2."""
    c = sf.lorenz_mean_z_bracket(degree=2)
    assert c.tier == sf.Tier.RIGOROUS and c.fail_p == 0.0
    assert abs((c.value - c.err) - 27.0) < 1e-12      # witness side exact
    assert (c.value + c.err) <= 28.0 + 1e-12          # <z> <= rho
    assert "sos" in c.provenance[0] and "exact" in c.provenance[0]
    assert "absorbing" in c.provenance[0]


def test_lorenz_z_quartic_tightens():
    """Degree-4 auxiliary polynomial, found by unrigorous float search,
    certified by the exact checker: the bound must strictly beat the
    quadratic rung."""
    c2 = sf.lorenz_mean_z_bracket(degree=2)
    c4 = sf.lorenz_mean_z_bracket(degree=4)
    assert c4.err < c2.err
    assert c4.tier == sf.Tier.RIGOROUS


def test_lorenz_bound_dominates_attractor():
    """Sanity from the other side: a long simulated trajectory's mean z
    (~23.5, chaotic attractor) must sit below the certified sup-over-
    trajectories bracket."""
    def f(x, y, z):
        return 10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z

    x, y, z = 1.0, 1.0, 1.0
    dt, n, acc = 0.004, 250_000, 0.0
    for i in range(n):
        ax, ay, az = f(x, y, z)
        bx, by, bz = f(x + dt / 2 * ax, y + dt / 2 * ay, z + dt / 2 * az)
        cx, cy, cz = f(x + dt / 2 * bx, y + dt / 2 * by, z + dt / 2 * bz)
        ex, ey, ez = f(x + dt * cx, y + dt * cy, z + dt * cz)
        x += dt / 6 * (ax + 2 * bx + 2 * cx + ex)
        y += dt / 6 * (ay + 2 * by + 2 * cy + ey)
        z += dt / 6 * (az + 2 * bz + 2 * cz + ez)
        acc += z
    mean_z = acc / n
    c4 = sf.lorenz_mean_z_bracket(degree=4)
    assert 22.0 < mean_z < 25.0                       # on the attractor
    assert mean_z <= c4.value + c4.err


def test_gw_surrogate_certified_mismatch():
    """The last TARGETS domain: a reduced-basis waveform surrogate with
    a distribution-free conformal certificate. 200 fresh parameter
    draws: the calibrated mismatch bound must hold at close to the
    declared rate, and the certificate must say EMPIRICAL, not more."""
    sur = sf.gw_surrogate_build(seed=7)
    rng = np.random.default_rng(1234)
    exceed, mms = 0, []
    for lam in rng.uniform(1.0, 2.0, 200):
        c = sf.gw_surrogate_eval(sur, float(lam))
        m = sf._gw_mismatch(c.value, sf._gw_chirp(float(lam)))
        mms.append(m)
        if m > sur["m_cal"]:
            exceed += 1
        assert c.tier == sf.Tier.EMPIRICAL
        assert c.fail_p == pytest.approx(1.0 / (sur["n_cal"] + 1))
        assert c.err == pytest.approx(math.sqrt(2.0 * sur["m_cal"]))
    # declared exceedance rate is 1/(n_cal+1) = 2%; allow sampling slack
    assert exceed <= 16
    assert np.median(mms) < sur["m_cal"]


def test_gw_surrogate_tightens_with_build_eps():
    """A tighter offline build must calibrate to a tighter certificate."""
    loose = sf.gw_surrogate_build(seed=7, eps_build=1e-3)
    tight = sf.gw_surrogate_build(seed=7, eps_build=1e-9)
    assert tight["m_cal"] < loose["m_cal"]
    assert len(tight["basis"]) > len(loose["basis"])


def test_gw_surrogate_refuses():
    """Outside the training hull, and below the calibrated mismatch:
    refusal, with the price of improvement named."""
    sur = sf.gw_surrogate_build(seed=7)
    with pytest.raises(ValueError, match="hull"):
        sf.gw_surrogate_eval(sur, 2.5)
    with pytest.raises(ValueError, match="calibrated"):
        sf.gw_surrogate_dispatch(sur, 1.5, tol=sur["m_cal"] / 10)
    c = sf.gw_surrogate_dispatch(sur, 1.5, tol=max(2 * sur["m_cal"], 1e-9))
    assert c.tier == sf.Tier.EMPIRICAL


def test_gw_mismatch_metric_sanity():
    """The mismatch functional itself: zero on itself, invariant under
    a global phase, and consistent with the 2-norm error statement
    err = sqrt(2 m) for normalized waveforms."""
    h = sf._gw_chirp(1.3)
    assert sf._gw_mismatch(h, h) == pytest.approx(0.0, abs=1e-12)
    assert sf._gw_mismatch(h, np.exp(1j * 0.7) * h) == \
        pytest.approx(0.0, abs=1e-12)
    g = sf._gw_chirp(1.31)
    m = sf._gw_mismatch(h, g)
    hn, gn = h / np.linalg.norm(h), g / np.linalg.norm(g)
    best = min(np.linalg.norm(hn - np.exp(1j * th) * gn)
               for th in np.linspace(0, 2 * math.pi, 20000))
    assert math.sqrt(2 * m) == pytest.approx(best, rel=1e-4)


def test_gci_extrapolate():
    """Grid-convergence certificate (the engineering V&V practice, made
    refusal-capable): a clean h^p ladder certifies and contains the
    h->0 limit; a ladder with no asymptotic range refuses."""
    hs = [0.4, 0.2, 0.1, 0.05]
    vals = [3.0 + 2.1 * h ** 1.7 for h in hs]
    c = sf.gci_extrapolate(vals, hs)
    assert c.tier == sf.Tier.EMPIRICAL
    assert abs(c.value - 3.0) <= c.err
    assert "order" in c.provenance[0] and "measured" in c.provenance[0]
    with pytest.raises(ValueError, match="asymptotic range"):
        sf.gci_extrapolate([3.0, 3.4, 3.1, 3.5], hs)


def test_sph_physics_sane():
    """The declared SPH model behaves like water: the dam-break front
    advances monotonically at a physical speed, and the wall feels
    nothing until the bore arrives."""
    out = sf.sph_dam_break(nres=16, snapshots=(1.0, 1.5, 2.0))
    fronts = [s[1].max() for s in out["snaps"]]
    assert fronts[0] < fronts[1] < fronts[2]
    speed = (fronts[2] - fronts[0]) / 1.0
    assert 0.9 < speed < 2.2          # Ritter dry-front speed is 2
    ts, F = out["ts"], out["F"]
    assert np.max(np.abs(F[ts < 1.5])) < 0.05
    assert np.max(F) > 0.3            # and the impact is a real event


def test_sph_hydrostatic_pressure():
    """A settled full-width column: fluid pressure near the floor must
    approach rho g depth."""
    out = sf.sph_dam_break(nres=16, tank=(1.5, 1.2), column=(1.5, 0.6),
                           T=2.5, alpha=0.5)
    fl = out["fluid"]
    pj = out["B"] * ((out["rho"][fl]) ** out["gamma"] - 1.0)
    low = out["py"][fl] < 0.12
    p_bot = float(np.mean(pj[low]))
    assert abs(p_bot - 0.55) < 0.2    # rho g (0.6 - 0.05) with SPH slop


def test_sph_impulse_certifies_peak_refuses():
    """The query triage, measured on a fixed-viscosity resolution
    ladder. The delivered impulse certifies (containment checked at a
    finer rung the certificate never saw) with an honestly wide err:
    the model is below its asymptotic range at this budget and the
    certificate says so. The raw peak force scatters with no
    asymptotic range at all, and the certifier refuses it — the
    honest verdict for breaking-wave peak loads."""
    J, peaks = {}, {}
    for n in (16, 24, 36, 48):
        out = sf.sph_dam_break(nres=n)
        J[n] = float(np.sum(out["F"]) * (out["ts"][1] - out["ts"][0]))
        peaks[n] = float(np.max(out["F"]))
    hs = [1 / 16, 1 / 24, 1 / 36]
    c = sf.gci_extrapolate([J[n] for n in (16, 24, 36)], hs)
    assert c.tier == sf.Tier.EMPIRICAL
    assert abs(J[48] - c.value) <= c.err
    assert c.err < 5.0 * abs(c.value)     # wide, and honestly so
    with pytest.raises(ValueError, match="asymptotic range"):
        sf.gci_extrapolate([peaks[n] for n in (16, 24, 36)], hs)


def test_sph_berm_design_brackets():
    """The engineering answer has three regimes, and the certificates
    tell them apart: a tall berm certifiably zeroes the delivered
    impulse (ladder constant at zero); an under-resolved intermediate
    berm REFUSES rather than pretend; the bare wall certifies a
    nonzero impulse (previous test)."""
    hs = [1 / 12, 1 / 18, 1 / 27]
    tall = [sf.sph_wall_impulse(n, obstacle=(2.9, 0.4, 0.4))
            for n in (12, 18, 27)]
    c = sf.gci_extrapolate(tall, hs)
    assert c.value == 0.0 and c.err == 0.0
    mid = [sf.sph_wall_impulse(n, obstacle=(2.9, 0.4, 0.12))
           for n in (12, 18, 27)]
    with pytest.raises(ValueError, match="asymptotic range"):
        sf.gci_extrapolate(mid, hs)


def test_sph_deterministic():
    """No randomness anywhere: two runs are bitwise identical."""
    a = sf.sph_dam_break(nres=12, T=1.0)
    b = sf.sph_dam_break(nres=12, T=1.0)
    assert np.array_equal(a["F"], b["F"])
