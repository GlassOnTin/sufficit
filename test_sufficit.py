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
    for n in (8000, 20000):
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
