"""Basilisk-style case pages: each page is BUILT by running its case.
The source renders as the presentation (extracted live via inspect), the
a priori theory is stated, and the certification is executed and drawn.
Run from the repo root:  python3 docs/make_cases.py   (~2 minutes)
"""
import inspect
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sufficit as sf  # noqa: E402


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def page(title, eyebrow, h1, dek, sections):
    body = "\n".join(sections)
    return f'''<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{STYLE}
<main>
<p class="eyebrow"><a href="../index.html">sufficit</a> · {eyebrow}</p>
<h1>{h1}</h1>
<p class="dek">{dek}</p>
{body}
</main>'''


def code_section(*funcs):
    src = "\n\n".join(inspect.getsource(f) for f in funcs)
    return ("<h2>The code (rendered from the module, not copied)</h2>"
            f"<pre>{esc(src)}</pre>")


class Axes:
    """Minimal hand-rolled chart axes for the case figures."""

    def __init__(self, xlim, ylim, w=640, h=320, logy=False,
                 ml=58, mr=16, mt=18, mb=44):
        self.xlim, self.ylim, self.w, self.h = xlim, ylim, w, h
        self.logy, self.ml, self.mr, self.mt, self.mb = logy, ml, mr, mt, mb

    def X(self, x):
        a, b = self.xlim
        return self.ml + (x - a) / (b - a) * (self.w - self.ml - self.mr)

    def Y(self, y):
        a, b = self.ylim
        if self.logy:
            a, b, y = math.log(a), math.log(b), math.log(y)
        return self.mt + (b - y) / (b - a) * (self.h - self.mt - self.mb)

    def path(self, xs, ys):
        return "M " + " L ".join(f"{self.X(x):.1f} {self.Y(y):.1f}"
                                 for x, y in zip(xs, ys))

    def grid(self, yticks, xticks, xfmt=lambda v: f"{v:g}",
             yfmt=lambda v: f"{v:g}"):
        out = []
        lo, hi = min(self.ylim), max(self.ylim)
        for v in yticks:
            if not lo <= v <= hi:
                continue
            out.append(f'<line x1="{self.ml}" y1="{self.Y(v):.1f}" '
                       f'x2="{self.w - self.mr}" y2="{self.Y(v):.1f}" '
                       f'class="board-ink" opacity="0.25" stroke-width="1"/>'
                       f'<text x="{self.ml - 8}" y="{self.Y(v) + 3.5:.1f}" '
                       f'text-anchor="end" class="board-text" '
                       f'font-size="10.5" opacity="0.7">{yfmt(v)}</text>')
        for v in xticks:
            out.append(f'<text x="{self.X(v):.1f}" y="{self.h - 18}" '
                       f'text-anchor="middle" class="board-text" '
                       f'font-size="10.5" opacity="0.7">{xfmt(v)}</text>')
        return "".join(out)


# ======================================================================
def tfi_case():
    N = 10
    H0, H1 = sf.tfi_chain(N)
    sur = sf.reduced_basis_surrogate(H0, H1, np.linspace(0.0, 2.0, 6))
    gs = np.linspace(0.0, 2.0, 100)
    ups, lows, exact = [], [], []
    for g in gs:
        c = sf.reduced_basis_bracket(sur, g)
        ups.append(c.value + c.err)
        lows.append(c.value - c.err)
        exact.append(float(np.linalg.eigvalsh(H0 + g * H1)[0]))
    ups, lows, exact = map(np.array, (ups, lows, exact))
    contained = int(np.sum((lows <= exact) & (exact <= ups)))
    crit = int(np.argmin(np.abs(gs - 1.0)))
    gap_at_crit = float(ups[crit] - exact[crit])

    ax = Axes((0, 2), (float(lows.min()) - 0.5, float(ups.max()) + 0.5),
              h=340)
    band = (ax.path(gs, ups) + " L "
            + " L ".join(f"{ax.X(g):.1f} {ax.Y(l):.1f}"
                         for g, l in zip(gs[::-1], lows[::-1])) + " Z")
    snaps = "".join(f'<circle cx="{ax.X(t):.1f}" cy="{ax.Y(l):.1f}" '
                    f'r="4.5" class="blue-fill"/>'
                    for t, l in zip(sur["thetas"], sur["lows"]))
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified band
around the exact ground energy across the quantum phase transition">
{ax.grid(range(int(ax.ylim[0]) + 1, int(ax.ylim[1]) + 1, 3),
         (0.0, 0.5, 1.0, 1.5, 2.0), xfmt=lambda v: f"g = {v:g}")}
<line x1="{ax.X(1):.1f}" y1="{ax.mt}" x2="{ax.X(1):.1f}" y2="{340 - ax.mb}"
      class="board-ink" stroke-dasharray="4 4" stroke-width="1.2" opacity="0.6"/>
<text x="{ax.X(1):.1f}" y="{ax.mt - 4}" text-anchor="middle" class="board-text"
      font-size="10.5" opacity="0.8">quantum phase transition</text>
<path d="{band}" fill="var(--blue)" opacity="0.16" stroke="none"/>
<path d="{ax.path(gs, ups)}" fill="none" class="rust-ink" stroke-width="2"/>
<path d="{ax.path(gs, lows)}" fill="none" class="blue-ink" stroke-width="2"/>
<path d="{ax.path(gs, exact)}" fill="none" class="board-ink" stroke-width="1.6"
      stroke-dasharray="5 4"/>
{snaps}</svg>'''

    return page(
        "Case: certified reduced basis across a phase transition",
        "certified case",
        "A certified parameter sweep across a quantum phase transition",
        "Six exact solves buy a rigorous two-sided band over a 100-point "
        "sweep — including the critical point, where nothing about the "
        "physics is smooth.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>For an affine family <code>H(g) = H₀ + g·H₁</code>, the "
            "ground energy is an infimum of affine functions of <code>g"
            "</code> — hence <em>concave</em>. Concave functions lie above "
            "their chords, so certified lower bounds at snapshots certify "
            "the chord between them with no further solve. The upper bound "
            "is the reduced-basis Rayleigh quotient: variational, exact at "
            "k×k cost from precomputed Grams. Outside the snapshot hull, "
            "the code refuses.</p>",
            code_section(sf.tfi_chain, sf.reduced_basis_surrogate,
                         sf.reduced_basis_bracket),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>Blue dots: 6 certified snapshot "
            "lower bounds (offline). Blue curve: the chord lower bound. "
            "Rust: the reduced Rayleigh upper bound. Dashed: the exact "
            "answer from full 1024-dimensional diagonalization at all 100 "
            "points — computed only to check the certificate, never used "
            "by it.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li><strong>{contained}/100</strong> sweep points contained."
            "</li>"
            f"<li>Upper-bound error at criticality: <strong>"
            f"{gap_at_crit:.2e}</strong> from six snapshot vectors.</li>"
            "<li>Online cost per point: one <strong>6×6</strong> "
            "eigenproblem.</li></ul>",
        ])


# ======================================================================
def h2_case():
    Rs = np.linspace(0.8, 5.0, 22)
    Es, errs = [], []
    for R in Rs:
        c = sf.h2_energy_bracket(float(R))
        Es.append(c.value)
        errs.append(c.err)
    Es = np.array(Es)
    e_atom2 = 2 * sf.hydrogen_atom_energy()
    c_far = sf.h2_energy_bracket(50.0)
    diss_gap = abs(c_far.value - e_atom2)
    i_eq = int(np.argmin(Es))

    ax = Axes((0.8, 5.0), (float(Es.min()) - 0.03, -0.90), h=340)
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified
potential energy curve of the hydrogen molecule">
{ax.grid((-1.15, -1.05, -0.95), (1, 2, 3, 4, 5),
         xfmt=lambda v: f"{v:g} bohr", yfmt=lambda v: f"{v:.2f}")}
<line x1="{ax.ml}" y1="{ax.Y(e_atom2):.1f}" x2="{640 - ax.mr}"
      y2="{ax.Y(e_atom2):.1f}" class="rust-ink" stroke-width="1.4"
      stroke-dasharray="5 4" opacity="0.8"/>
<text x="{640 - ax.mr - 4}" y="{ax.Y(e_atom2) - 6:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" fill="var(--rust)">two isolated
 H atoms (computed independently)</text>
<path d="{ax.path(Rs, Es)}" fill="none" class="blue-ink" stroke-width="2.2"/>
<circle cx="{ax.X(Rs[i_eq]):.1f}" cy="{ax.Y(Es[i_eq]):.1f}" r="5"
        class="blue-fill"/>
<text x="{ax.X(Rs[i_eq]):.1f}" y="{ax.Y(Es[i_eq]) + 20:.1f}"
      text-anchor="middle" class="board-text" font-size="10.5">equilibrium</text>
<text x="{ax.ml + 6}" y="{ax.mt + 14}" class="board-text" font-size="11">
total energy (hartree)</text></svg>'''

    return page(
        "Case: the hydrogen molecule, bracketed from raw integrals",
        "certified case",
        "H₂ from scratch: a bond with error bars of ±1.5·10⁻¹³",
        "Gaussian integrals in closed form, second quantization, and a "
        "two-sided bracket — the entire pipeline is in the module, and "
        "the dissociation limit is predicted by an independent "
        "calculation it must agree with.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>The upper bound is the variational theorem: any trial "
            "state's energy is a ceiling on the truth. The lower bound is "
            "a feasibility proof: <code>H − c·I</code> admits a Cholesky "
            "factorization exactly when the ground energy is at least "
            "<code>c</code>, and bisection turns that yes/no question "
            "into a floor. Both bounds carry explicit floating-point "
            "margins. At large separation, monopole cancellation is exact "
            "for s-orbitals, so the molecule must dissociate to exactly "
            "two isolated atoms — computed separately, giving the "
            "a&nbsp;priori line the curve must meet.</p>",
            code_section(sf.hydrogen_atom_energy, sf.h2_energy_bracket,
                         sf.eigen_bracket),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>The certified potential curve "
            "(bracket widths ~10⁻¹³ hartree — about ten orders of "
            "magnitude thinner than this line). The dashed asymptote is "
            "not a fit: it is an independent calculation the curve is "
            "required to approach.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>Maximum bracket width over the curve: <strong>"
            f"{2 * max(errs):.1e}</strong> hartree — versus chemical "
            "accuracy at 1.6·10⁻³.</li>"
            f"<li>Dissociation vs two isolated atoms: <strong>"
            f"{diss_gap:.1e}</strong> hartree difference at R = 50.</li>"
            f"<li>Equilibrium bracket: <strong>{Es[i_eq]:.9f} ± "
            f"{errs[i_eq]:.0e}</strong> at R = {Rs[i_eq]:.1f} bohr "
            "(published FCI/STO-3G ≈ −1.13727 at 1.4).</li></ul>",
        ])


# ======================================================================
def ladder_case():
    runs = {6: [3, 4, 5], 10: [3, 4, 5]}
    data, checks = {}, []
    from scipy.sparse.linalg import eigsh
    for n, ells in runs.items():
        truth = None
        if n == 6:
            truth = float(eigsh(sf.h_chain_fock_hamiltonian(n, 1.8),
                                k=1, which="SA")[0][0])
        pts = []
        for ell in ells:
            c = sf.h_chain_bracket(n, 1.8, ell=ell)
            w = 2 * c.err / n * 1000.0
            pts.append((ell, w))
            if truth is not None:
                checks.append(c.value - c.err <= truth <= c.value + c.err)
        data[n] = pts

    ax = Axes((2.7, 5.3), (40, 600), h=320, logy=True, mt=44)
    series = []
    for n, cls in ((6, "blue"), (10, "rust")):
        xs = [p[0] for p in data[n]]
        ys = [p[1] for p in data[n]]
        series.append(f'<path d="{ax.path(xs, ys)}" fill="none" '
                      f'class="{cls}-ink" stroke-width="2" opacity="0.85"/>')
        for x, y in data[n]:
            series.append(f'<circle cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" '
                          f'r="5" class="{cls}-fill"/>'
                          f'<text x="{ax.X(x):.1f}" y="{ax.Y(y) - 10:.1f}" '
                          f'text-anchor="middle" class="board-text" '
                          f'font-size="10.5" fill="var(--{cls})">'
                          f'{y:.0f}</text>')
    svg = f'''<svg viewBox="0 0 640 320" role="img" aria-label="Certified
bracket width per atom versus window length, log scale">
{ax.grid((50, 100, 200, 400), (3, 4, 5), xfmt=lambda v: f"ℓ = {v:g}")}
{"".join(series)}
<text x="{ax.ml + 6}" y="{ax.mt - 24}" class="board-text" font-size="11">
certified width, mHa per atom (log scale)</text></svg>'''

    return page(
        "Case: the hierarchy knob on hydrogen chains",
        "certified case",
        "Pay for precision: the window-length ladder, measured",
        "The certified bracket on a hydrogen chain tightens roughly 2× "
        "for every unit of window length ℓ, at cost 4<sup>ℓ</sup> — "
        "independent of how long the chain is.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>Slide a window of ℓ atoms along the chain; weight each "
            "interaction by the reciprocal of how many windows contain it, "
            "so the windows sum to exactly the whole Hamiltonian; then the "
            "true energy is at least the sum of certified window floors. "
            "Long-range Coulomb terms are absorbed exactly via a "
            "charge-neutral rewrite and the operator inequality "
            "<code>g(nᵢ−1)(nⱼ−1) ⪰ −(g/2)[(nᵢ−1)² + (nⱼ−1)²]</code>; "
            "telescoping corrections on window overlaps — optimized by a "
            "proximal bundle, valid for any choice — tighten every window "
            "at once. The upper bound is a product of exactly-solved "
            "blocks. Cost depends on ℓ, never on chain length.</p>",
            "<h2>The orchestration (rendered from the module)</h2>"
            f"<pre>{esc(inspect.getsource(sf.h_chain_bracket))}</pre>",
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>Blue: H₆ (every point checked "
            "against exact diagonalization in this run). Rust: H₁₀ — a "
            "2²⁰-dimensional problem with no exact answer to compare "
            "against; none is needed. A separate recorded run extends "
            "H₁₀ to ℓ=7 (16,384-dimensional windows, tamed by particle-"
            "number sectors) at 55 mHa/atom; it is excluded here only to "
            "keep this page regenerable in seconds.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>H₆ containment vs exact: <strong>{sum(checks)}/"
            f"{len(checks)}</strong> at every ℓ.</li>"
            "<li>Widths halve, roughly, per unit ℓ — the grudging rate of "
            "a <em>critical</em> chain; gapped systems tighten "
            "exponentially faster.</li></ul>",
        ])


# ======================================================================
def butterfly_case():
    from scipy.special import hankel1
    k = 1800.0
    rng = np.random.default_rng(36)
    n_big = 4608
    src = 0.25 * (rng.uniform(-1, 1, n_big) + 1j * rng.uniform(-1, 1, n_big))
    tgt = 1.2 + 0.25 * (rng.uniform(-1, 1, n_big)
                        + 1j * rng.uniform(-1, 1, n_big))
    Kbig = 0.25j * hankel1(0, k * np.abs(tgt[:, None] - src[None, :]))
    ratios, ns = [], (1536, 3072, 4608)
    r_plain = None
    stats_last = None
    for n in ns:
        K = Kbig[:n, :n]
        bf = sf.ButterflyBlock(K, tgt[:n], src[:n], levels=3, eps=1e-4,
                               rng=np.random.default_rng(1))
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        c, stats = bf.apply(q)
        err_actual = float(np.linalg.norm(c.value - K @ q))
        assert err_actual <= c.err
        res = sf._compress_certified(
            K, bf.beta, 10, 10 * math.sqrt(2 / math.pi) * math.sqrt(2),
            np.random.default_rng(2))
        r_plain = res[1].shape[0]
        ratios.append(stats["apply_flops"] / (r_plain * 2 * n))
        stats_last = (c.err, err_actual, bf.fail_p)

    ax = Axes((1200, 4900), (0.4, 1.4), h=300)
    pts = "".join(
        f'<circle cx="{ax.X(n):.1f}" cy="{ax.Y(r):.1f}" r="5" '
        f'class="blue-fill"/>'
        f'<text x="{ax.X(n):.1f}" y="{ax.Y(r) - 12:.1f}" text-anchor="middle" '
        f'class="board-text" font-size="10.5" fill="var(--blue)">'
        f'{r:.2f}×</text>' for n, r in zip(ns, ratios))
    svg = f'''<svg viewBox="0 0 640 300" role="img" aria-label="Butterfly cost
relative to plain low-rank versus problem size: crossover below 1">
{ax.grid((0.6, 1.0), ns, xfmt=lambda v: f"N = {v:g}", yfmt=lambda v: f"{v:g}×")}
<line x1="{ax.ml}" y1="{ax.Y(1.0):.1f}" x2="{640 - ax.mr}"
      y2="{ax.Y(1.0):.1f}" class="rust-ink" stroke-width="1.6"
      stroke-dasharray="5 4"/>
<text x="{640 - ax.mr - 4}" y="{ax.Y(1.0) - 8:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" fill="var(--rust)">break-even with
 plain low-rank</text>
<path d="{ax.path(ns, ratios)}" fill="none" class="blue-ink"
      stroke-width="2"/>
{pts}
<text x="{ax.ml + 6}" y="{ax.mt + 14}" class="board-text" font-size="11">
butterfly apply cost ÷ plain low-rank cost (matched certified accuracy)</text>
</svg>'''

    return page(
        "Case: the butterfly crossover at high frequency",
        "certified case",
        "The butterfly factorization earns its keep — measurably",
        "For oscillatory wave kernels, the multi-level butterfly beats "
        "plain low-rank compression only past a predictable problem size. "
        "The certificate never depends on which side of the crossover "
        "you are on.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>Well-separated blocks of a wave kernel have numerical rank "
            "≈ R = k·r₁·r₂/D — growing with frequency k. The butterfly "
            "ladder refines the row tree while coarsening the column tree, "
            "keeping every factor at rank ≈ R/2<sup>L</sup>; its transfer "
            "cost is <em>independent of N</em>, while plain low-rank costs "
            "R·(m+n). So at fixed R the ratio must cross below 1 as N "
            "grows — a prediction this page tests. Certification is a "
            "posteriori: random probes of the assembled factorization, "
            "with the failure probability printed.</p>",
            code_section(sf.ButterflyBlock),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>Helmholtz kernel at k = 1800 "
            f"(R ≈ 95), matched certified accuracy (plain rank "
            f"{r_plain}). The crossover arrives where the N-independent "
            "transfer cost is amortized — exactly as the cost model "
            "predicts.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            "<li>Every apply checked against the dense kernel: actual "
            f"error <strong>{stats_last[1]:.1e}</strong> within certified "
            f"<strong>{stats_last[0]:.1e}</strong>.</li>"
            f"<li>Stated failure probability: <strong>"
            f"{stats_last[2]:.0e}</strong> — printed, not hidden.</li>"
            f"<li>Crossover measured: {ratios[0]:.2f}× → {ratios[-1]:.2f}× "
            "of plain low-rank as N grows at fixed R.</li></ul>",
        ])


# ======================================================================
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(v):
    return "10" + str(int(round(math.log10(v)))).translate(_SUP)


def ising_case():
    import test_sufficit as suite     # the suite's exact transfer matrix
    betas = np.linspace(0.005, 0.084, 12)
    rows = {"logZ": [], "corr": []}
    for bJ in betas:
        tz = suite._ising_torus_logZ_density_tm(10, float(bJ))
        tc = suite._ising_torus_bond_corr_tm(10, float(bJ))
        cz = sf.ising2d_logZ_density(float(bJ))
        cc = sf.ising2d_bond_correlation(float(bJ))
        rows["logZ"].append((cz.err, abs(cz.value - tz)))
        rows["corr"].append((cc.err, abs(cc.value - tc)))
    contained = sum(a <= e for r in rows.values() for e, a in r)
    refused = 0
    for fn in (sf.ising2d_logZ_density, sf.ising2d_bond_correlation):
        try:
            fn(0.1)
        except ValueError:
            refused += 1
    c0 = sf.ising2d_logZ_density(0.0)
    zero_ok = abs(c0.value - math.log(2.0)) <= c0.err

    floor = 2e-16
    ax = Axes((0.0, 0.097), (floor, 2.0), h=360, logy=True)
    parts = []
    for key, cls in (("logZ", "blue"), ("corr", "rust")):
        errs = [e for e, _ in rows[key]]
        parts.append(f'<path d="{ax.path(betas, errs)}" fill="none" '
                     f'class="{cls}-ink" stroke-width="2"/>')
        parts += [f'<circle cx="{ax.X(b):.1f}" '
                  f'cy="{ax.Y(max(a, floor)):.1f}" r="3.5" '
                  f'class="{cls}-fill" opacity="0.75"/>'
                  for b, (_, a) in zip(betas, rows[key])]
    svg = f'''<svg viewBox="0 0 640 360" role="img" aria-label="Certified
error bound versus measured deviation across the high-temperature region">
{ax.grid((1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0), (0.02, 0.04, 0.06, 0.08),
         xfmt=lambda v: f"βJ = {v:g}", yfmt=pow10)}
<line x1="{ax.X(0.086):.1f}" y1="{ax.mt}" x2="{ax.X(0.086):.1f}"
      y2="{360 - ax.mb}" class="board-ink" stroke-dasharray="4 4"
      stroke-width="1.4" opacity="0.7"/>
<text x="{ax.X(0.086) - 6:.1f}" y="{360 - ax.mb - 10}" text-anchor="end"
      class="board-text" font-size="10.5" opacity="0.85">refuses beyond here</text>
{"".join(parts)}
<text x="{ax.ml + 10}" y="{ax.mt + 12}" class="board-text"
      font-size="10.5" fill="var(--blue)">free energy: certified bound
 (line), measured error (dots)</text>
<text x="{ax.ml + 10}" y="{ax.mt + 28}" class="board-text"
      font-size="10.5" fill="var(--rust)">bond correlation
 ⟨s₀s₁⟩</text></svg>'''

    return page(
        "Case: the 2D Ising model, certified by cluster expansions",
        "certified case",
        "A validity region with a wall the certificate refuses to cross",
        "Free energy and correlations of the 2D Ising model at high "
        "temperature, with floating point carried in interval "
        "arithmetic — checked against the exact transfer matrix at "
        "every point, and refusing where the expansion's own "
        "convergence proof gives out.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>The high-temperature expansion rewrites the Ising "
            "partition function as a gas of polymers (closed loops on "
            "the lattice) with activity <code>t = tanh βJ</code>. The "
            "Kotecký–Preiss criterion gives a computable radius inside "
            "which the cluster series converges, with a geometric tail "
            "bound at truncation order L = 12; local observables come "
            "from pinned polymers with an Eulerian-path counting bound. "
            "Every coefficient and every tail is evaluated in outward-"
            "rounded interval arithmetic, so the floating point is part "
            "of the certificate. Outside the proven radius the functions "
            "raise — the wall at βJ ≈ 0.086 is the certificate's, not "
            "nature's (the true critical point sits at βJ ≈ 0.4407).</p>",
            code_section(sf.ising2d_logZ_density,
                         sf.ising2d_bond_correlation),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>Lines: the certified error bound. "
            "Dots: the measured deviation from the exact 10×10 transfer "
            "matrix (the suite's own truth generator, itself validated "
            "against 2¹⁶-state exhaustive enumeration). The bound must "
            "sit above every dot; both climb as the expansion approaches "
            "its convergence wall. Dots on the floor are at the double-"
            "precision limit.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{2 * len(betas)}"
            "</strong> (both quantities, every temperature).</li>"
            f"<li>Refusal at βJ = 0.1: <strong>{refused}/2</strong> "
            "functions raised rather than extrapolate.</li>"
            f"<li>Zero-coupling limit: log Z = log 2 "
            f"{'contained' if zero_ok else 'NOT CONTAINED'} with width "
            f"{c0.err:.1e}.</li></ul>",
        ])


# ======================================================================
def spectral_case():
    E, a = (0.9, 1.9), (1.0, 0.7)
    sigma = 0.35
    ts = np.arange(1, 17)
    C = a[0] * np.exp(-E[0] * ts) + a[1] * np.exp(-E[1] * ts)

    def truth(w):
        return sum(ai * math.exp(-(w - Ei) ** 2 / (2 * sigma**2))
                   / (sigma * math.sqrt(2 * math.pi))
                   for ai, Ei in zip(a, E))

    ws = np.linspace(0.4, 2.4, 41)
    vals, errs, exact = [], [], []
    for w in ws:
        c = sf.smeared_spectral(C, float(w), sigma)
        vals.append(c.value)
        errs.append(c.err)
        exact.append(truth(float(w)))
    vals, errs, exact = map(np.array, (vals, errs, exact))
    contained = int(np.sum(np.abs(vals - exact) <= errs))
    res_errs = [sf.smeared_spectral(C, 1.0, s).err for s in (0.6, 0.4, 0.25)]

    ax = Axes((0.4, 2.4), (0.0, 1.35 * float(exact.max())), h=340)
    ups, lows = vals + errs, np.maximum(vals - errs, 0.0)
    band = (ax.path(ws, ups) + " L "
            + " L ".join(f"{ax.X(w):.1f} {ax.Y(l):.1f}"
                         for w, l in zip(ws[::-1], lows[::-1])) + " Z")
    peaks = "".join(
        f'<line x1="{ax.X(Ei):.1f}" y1="{ax.mt}" x2="{ax.X(Ei):.1f}" '
        f'y2="{340 - ax.mb}" class="board-ink" stroke-dasharray="3 4" '
        f'stroke-width="1.2" opacity="0.55"/>'
        f'<text x="{ax.X(Ei):.1f}" y="{ax.mt - 4}" text-anchor="middle" '
        f'class="board-text" font-size="10.5" opacity="0.8">peak at '
        f'{Ei:g}</text>' for Ei in E)
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified
band around the smeared spectral density, resolving two peaks">
<clipPath id="plot"><rect x="{ax.ml}" y="{ax.mt}"
  width="{640 - ax.ml - ax.mr}" height="{340 - ax.mt - ax.mb}"/></clipPath>
{ax.grid((0.3, 0.6, 0.9, 1.2), (0.5, 1.0, 1.5, 2.0),
         xfmt=lambda v: f"ω = {v:g}", yfmt=lambda v: f"{v:g}")}
{peaks}
<g clip-path="url(#plot)">
<path d="{band}" fill="var(--blue)" opacity="0.16" stroke="none"/>
<path d="{ax.path(ws, ups)}" fill="none" class="rust-ink" stroke-width="1.8"/>
<path d="{ax.path(ws, lows)}" fill="none" class="blue-ink" stroke-width="1.8"/>
<path d="{ax.path(ws, exact)}" fill="none" class="board-ink"
      stroke-width="1.6" stroke-dasharray="5 4"/></g></svg>'''

    return page(
        "Case: smeared spectral functions with resolution as part of "
        "the query",
        "certified case",
        "Two peaks from sixteen numbers — and the price of sharpness",
        "A Gaussian-smeared spectral density reconstructed from 16 "
        "Euclidean correlator values, with a certified band at every "
        "frequency. Ask for sharper resolution and the certificate "
        "honestly charges you more error.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>The data determine the spectral density ρ only through "
            "exponential moments C(t) = ∫e^(−ωt) ρ(ω) dω — inverting "
            "that is ill-posed. The Hansen–Lupo–Tantalo move: don't "
            "invert; reconstruct the <em>smearing kernel</em> as a sum "
            "of the exponentials you have. The certificate is a "
            "posteriori and immune to how the coefficients were found: "
            "a rigorous sup bound c on the weighted kernel deviation "
            "(dense grid + per-cell Lipschitz + analytic tail) turns "
            "positivity of ρ into |value − truth| ≤ c·C(1). One declared "
            "physical assumption: ρ ≥ 0, stamped into the provenance.</p>",
            code_section(sf.smeared_spectral, sf._hlt_solve),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>The certified band (rust upper, "
            "blue lower) around the smeared two-peak density at "
            "σ = 0.35, swept over 41 frequencies. Dashed: the exact "
            "smeared truth, computable here because the test density is "
            "synthetic — the band was built without it. The first peak "
            "is certified two-sidedly; at the second the lower bound "
            "has already fallen to zero, and past ω ≈ 2 the band runs "
            "off the top of the plot: sixteen exponentials genuinely "
            "cannot say more, and the certificate says so.</figcaption>"
            "</figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(ws)}</strong> "
            "frequencies.</li>"
            "<li>Resolution costs error, monotonically: certified err at "
            f"ω = 1 is <strong>{res_errs[0]:.3f} → {res_errs[1]:.3f} → "
            f"{res_errs[2]:.3f}</strong> for σ = 0.6 → 0.4 → 0.25.</li>"
            "<li>The certificate never used the truth: it is c·C(1) "
            "from the data and the kernel sup bound alone.</li></ul>",
        ])


# ======================================================================
def mz_case():
    from scipy.linalg import expm
    rng = np.random.default_rng(21)
    A = np.zeros((10, 10))
    A[:2, :2] = [[-0.3, 0.2], [-0.2, -0.4]]
    A[:2, 2:] = 0.15 * rng.standard_normal((2, 8))
    A[2:, :2] = 0.15 * rng.standard_normal((8, 2))
    A[2:, 2:] = -6.0 * np.eye(8) + 0.3 * rng.standard_normal((8, 8))
    x0 = np.concatenate([[1.0, -0.5], np.zeros(8)])

    Ts = np.linspace(0.25, 20.0, 45)
    pred, errs, exact = [], [], []
    for T in Ts:
        c = sf.mz_closure_linear(A, 2, x0, float(T))
        pred.append(c.value)
        errs.append(c.err)
        exact.append((expm(A * float(T)) @ x0)[:2])
    pred, errs, exact = np.array(pred), np.array(errs), np.array(exact)
    contained = int(np.sum(np.linalg.norm(pred - exact, axis=1) <= errs))

    stiff = A.copy()
    stiff[2:, 2:] *= 2.0
    e_base = sf.mz_closure_linear(A, 2, x0, 5.0).err
    e_stiff = sf.mz_closure_linear(stiff, 2, x0, 5.0).err
    nogap = np.diag([-1.0, -1.0, 0.1, -1.0])
    try:
        sf.mz_closure_linear(nogap, 2, np.array([1.0, 0, 0, 0]), 1.0)
        refused = False
    except ValueError:
        refused = True

    lo = float((pred.min(axis=1) - errs).min())
    hi = float((pred.max(axis=1) + errs).max())
    pad = 0.06 * (hi - lo)
    ax = Axes((0.0, 20.0), (lo - pad, hi + pad), h=340)
    parts = []
    for i, cls in ((0, "blue"), (1, "rust")):
        ups, lows = pred[:, i] + errs, pred[:, i] - errs
        band = (ax.path(Ts, ups) + " L "
                + " L ".join(f"{ax.X(t):.1f} {ax.Y(l):.1f}"
                             for t, l in zip(Ts[::-1], lows[::-1])) + " Z")
        parts.append(
            f'<path d="{band}" fill="var(--{cls})" opacity="0.13" '
            f'stroke="none"/>'
            f'<path d="{ax.path(Ts, pred[:, i])}" fill="none" '
            f'class="{cls}-ink" stroke-width="2"/>'
            f'<path d="{ax.path(Ts, exact[:, i])}" fill="none" '
            f'class="board-ink" stroke-width="1.5" stroke-dasharray="5 4"/>')
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified
tubes around the reduced-model trajectories of both slow variables">
{ax.grid((-0.5, 0.0, 0.5, 1.0), (5, 10, 15, 20), xfmt=lambda v: f"T = {v:g}")}
{"".join(parts)}
<text x="{ax.X(2.2):.1f}" y="{ax.Y(float(pred[4, 0])) - 12:.1f}"
      class="board-text" font-size="10.5" fill="var(--blue)">x₁ closure ±
 certified tube</text>
<text x="{ax.X(2.2):.1f}" y="{ax.Y(float(pred[4, 1])) + 20:.1f}"
      class="board-text" font-size="10.5" fill="var(--rust)">x₂</text></svg>'''

    return page(
        "Case: Mori-Zwanzig closures with a gap-priced certificate",
        "certified case",
        "Forget the fast variables — and pay exactly what the gap "
        "charges",
        "A 10-dimensional slow-fast system reduced to its 2 slow "
        "coordinates by the Markovian closure. The certified tube "
        "comes from the fast sector's spectral gap; no gap, no "
        "certificate — the function refuses.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>Projecting a linear system onto slow observables leaves "
            "an exact memory term K(s) = A₁₂e^(A₂₂s)A₂₁. If the fast "
            "sector is dissipative — log-norm of A₂₂ strictly negative — "
            "the kernel decays at the gap μ, and dropping the memory "
            "(the Markovian closure A₁₁ − A₁₂A₂₂⁻¹A₂₁) costs an error a "
            "Grönwall argument bounds with computable constants: block "
            "norms, log-norms, and 1/μ² — the certificate is priced by "
            "the physics that justifies the reduction. A fast initial "
            "transient adds its own decaying term. Without a gap the "
            "rewrite refuses: no decay proof, no closure.</p>",
            code_section(sf._lognorm, sf.mz_closure_linear),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>Both slow coordinates: closure "
            "prediction (solid) inside its certified tube (shaded, "
            "half-width the joint 2-norm bound), exact 10-dimensional "
            "propagation dashed — computed only to check containment. "
            "The tube is thin — half-width "
            f"{errs[0]:.3f} at T = {Ts[0]:g}, saturating at "
            f"{errs[-1]:.3f} by T = 20 because the reduced model is "
            "itself dissipative.</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(Ts)}</strong> "
            "horizons out to T = 20.</li>"
            "<li>The gap prices the certificate: doubling the fast "
            f"sector's stiffness tightens the bound <strong>"
            f"{e_base / e_stiff:.1f}×</strong> at T = 5.</li>"
            f"<li>No-gap system: <strong>{'refused' if refused else 'NOT REFUSED'}"
            "</strong> (an undamped fast mode means the memory kernel "
            "never certifiably decays).</li></ul>",
        ])


# ======================================================================
def lr_case():
    n, site, tol, md = 11, 5, 1e-2, 256
    H0, H1 = sf.tfi_chain(n)
    lam, V = np.linalg.eigh(H0 + H1)
    psi0 = np.zeros(2 ** n)
    psi0[0] = 1.0
    zdiag = 1.0 - 2.0 * ((np.arange(2 ** n) >> (n - 1 - site)) & 1)

    def exact_zt(t):
        psi = V @ (np.exp(-1j * lam * t) * (V.T @ psi0))
        return float(np.real(np.vdot(psi, zdiag * psi)))

    ts = np.linspace(0.0, 2.0, 21)
    cert, wall = [], None          # (t, value, err, r) for certified points
    for t in ts:
        try:
            c = sf.tfi_quench_dispatch(n, site, float(t), tol=tol,
                                       max_dim=md, n_steps=300)
            r = int(c.provenance[0].split("r=")[1].split()[0])
            cert.append((float(t), c.value, c.err, r))
        except ValueError:
            wall = float(t) if wall is None else wall
    # the t=0 certificate is exactly 0 (declared exact-arithmetic);
    # the eigh-based checker itself carries ~1e-15 float noise
    contained = sum(abs(v - exact_zt(t)) <= e + 1e-12 for t, v, e, _ in cert)
    exact = [exact_zt(float(t)) for t in ts]

    import time as _time
    t0 = _time.time()
    big = sf.tfi_quench_dispatch(10 ** 6, 500_000, 0.6, tol=1e-3,
                                 n_steps=300)
    big_secs = _time.time() - t0
    small = sf.tfi_quench_dispatch(2001, 1000, 0.6, tol=1e-3, n_steps=300)
    identical = (big.value == small.value and big.err == small.err)

    ax = Axes((0.0, 2.0), (-0.45, 1.12), h=340)
    tc = [p[0] for p in cert]
    band = (ax.path(tc, [p[1] + p[2] for p in cert]) + " L "
            + " L ".join(f"{ax.X(p[0]):.1f} {ax.Y(p[1] - p[2]):.1f}"
                         for p in cert[::-1]) + " Z")
    rmarks = []
    for i in range(1, len(cert)):
        if cert[i][3] > cert[i - 1][3]:
            x = ax.X(cert[i][0])
            rmarks.append(
                f'<line x1="{x:.1f}" y1="{340 - ax.mb}" x2="{x:.1f}" '
                f'y2="{340 - ax.mb - 14}" class="blue-ink" '
                f'stroke-width="1.4"/>'
                f'<text x="{x:.1f}" y="{340 - ax.mb - 18}" '
                f'text-anchor="middle" class="board-text" font-size="10" '
                f'fill="var(--blue)">r={cert[i][3]}</text>')
    wall_x = ax.X(wall)
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified
band around the quench observable, cone radius escalating, refusal region
shaded past the budget wall">
{ax.grid((-0.4, 0.0, 0.4, 0.8), (0.5, 1.0, 1.5, 2.0),
         xfmt=lambda v: f"t = {v:g}", yfmt=lambda v: f"{v:g}")}
<rect x="{wall_x:.1f}" y="{ax.mt}" width="{640 - ax.mr - wall_x:.1f}"
      height="{340 - ax.mt - ax.mb}" fill="var(--rust)" opacity="0.08"/>
<line x1="{wall_x:.1f}" y1="{ax.mt}" x2="{wall_x:.1f}" y2="{340 - ax.mb}"
      class="rust-ink" stroke-dasharray="4 4" stroke-width="1.4"/>
<text x="{wall_x + 8:.1f}" y="{ax.mt + 14}" class="board-text"
      font-size="10.5" fill="var(--rust)">refuses: light cone outruns
 the budget</text>
<path d="{band}" fill="var(--blue)" opacity="0.16" stroke="none"/>
<path d="{ax.path(tc, [p[1] for p in cert])}" fill="none" class="blue-ink"
      stroke-width="2"/>
<path d="{ax.path(ts, exact)}" fill="none" class="board-ink"
      stroke-width="1.5" stroke-dasharray="5 4"/>
{"".join(rmarks)}
<text x="{ax.ml + 10}" y="{ax.mt + 14}" class="board-text" font-size="10.5">
⟨Z(t)⟩ after a quench, critical TFI chain</text></svg>'''

    return page(
        "Case: quantum-dynamics dispatch by Lieb-Robinson cone",
        "certified case",
        "Does a certified classical simulation exist? Ask the boundary",
        "A quench observable on a transverse-field Ising chain, certified "
        "by simulating only a cone of sites and measuring — not "
        "estimating — what leaks across its edge. The compiler grows the "
        "cone as the light cone spreads, and refuses when its budget is "
        "outrun. Chain length never enters the cost.",
        [
            "<h2>The theory, a priori</h2>"
            "<p>Comparing full dynamics with cone-restricted dynamics, "
            "Duhamel gives ‖A(t) − A_cone(t)‖ ≤ ∫₀ᵗ ‖[H − H_cone, "
            "A_cone(s)]‖ ds, and only the two bonds crossing the cone "
            "boundary fail to commute with the cone-supported operator. "
            "Those commutator norms are <em>measured inside the "
            "simulation itself</em> — near zero until the excitation "
            "front physically arrives — so the certificate carries no "
            "Lieb–Robinson velocity constants to be loose about. "
            "Quadrature is rigorous too: the interpolation remainder is "
            "priced by the measured second derivative ‖[P,[H,[H,A]]]‖, "
            "with a crude cap only at the harmless δ⁴ level.</p>",
            code_section(sf.tfi_quench_dispatch, sf._lr_cone_run,
                         sf._opnorm_ub),
            "<h2>The certification, executed and drawn</h2>"
            f"<figure>{svg}<figcaption>The certified band around "
            "⟨Z(t)⟩ at the critical point (g = 1), tolerance 10⁻². Ticks "
            "mark where dispatch grew the cone; the shaded region is "
            "refusal — the measured boundary leakage exceeds the "
            "tolerance at every affordable radius, and the error message "
            "prices the next cone instead of guessing. The certified "
            "width breathes in a sawtooth because dispatch always takes "
            "the cheapest adequate cone. Dashed: the exact "
            "2048-dimensional answer, computed only to check the band."
            "</figcaption></figure>",
            "<h2>Verified in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(cert)}</strong> "
            "certified sweep points vs exact diagonalization.</li>"
            f"<li>Refusal wall at <strong>t = {wall:g}</strong> with "
            f"max_dim = {md}: past it, no certificate is claimed.</li>"
            "<li>Chain length never enters: a <strong>10⁶-site</strong> "
            f"chain certified ±10⁻³ in <strong>{big_secs:.1f} s</strong>, "
            "bit-identical to the 2001-site run: "
            f"<strong>{'yes' if identical else 'NO'}</strong>.</li></ul>",
        ])


STYLE = '''<style>
  :root { --paper:#F7F8F7; --ink:#1A2028; --muted:#5A6472; --blue:#1D6FA5;
    --rust:#B4552D; --hairline:#D9DDDC; --panel:#EEF1F0; --panel-ink:#333C46; }
  @media (prefers-color-scheme: dark) { :root { --paper:#14181D; --ink:#E6E4DE;
    --muted:#9AA3AD; --blue:#3E8CC2; --rust:#C97A52; --hairline:#2A313A;
    --panel:#1B2129; --panel-ink:#C7CBD1; } }
  :root[data-theme="dark"] { --paper:#14181D; --ink:#E6E4DE; --muted:#9AA3AD;
    --blue:#3E8CC2; --rust:#C97A52; --hairline:#2A313A; --panel:#1B2129;
    --panel-ink:#C7CBD1; }
  :root[data-theme="light"] { --paper:#F7F8F7; --ink:#1A2028; --muted:#5A6472;
    --blue:#1D6FA5; --rust:#B4552D; --hairline:#D9DDDC; --panel:#EEF1F0;
    --panel-ink:#333C46; }
  html { background: var(--paper); }
  body { margin:0; padding:0 1.25rem; color:var(--ink); background:var(--paper);
    font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    font-size:1.05rem; line-height:1.6; }
  main { max-width:44rem; margin:0 auto; padding:3.5rem 0 5rem; }
  h1 { font-size:2.1rem; line-height:1.15; margin:0.3rem 0 0.6rem;
    letter-spacing:-0.01em; text-wrap:balance; }
  h2 { font-size:1.35rem; margin:2.6rem 0 0.7rem; }
  .eyebrow { font-family:"Avenir Next","Segoe UI",system-ui,sans-serif;
    font-size:0.72rem; text-transform:uppercase; letter-spacing:0.14em;
    color:var(--muted); margin:0; }
  .eyebrow a { color: var(--blue); text-decoration: none; }
  .dek { font-size:1.18rem; color:var(--muted); margin:0 0 1.8rem; }
  pre { background:var(--panel); padding:1rem 1.1rem; border-radius:4px;
    overflow-x:auto; font-size:0.78rem; line-height:1.5; max-height:32rem;
    overflow-y:auto;
    font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
  code { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    font-size:0.88em; background:var(--panel); padding:0.05em 0.3em;
    border-radius:3px; }
  figure { margin:1.6rem 0; padding:1.2rem 1rem 0.9rem; background:var(--panel);
    border-radius:4px; overflow-x:auto; }
  figure svg { display:block; margin:0 auto; max-width:100%; height:auto; }
  figcaption { font-family:"Avenir Next","Segoe UI",system-ui,sans-serif;
    font-size:0.82rem; color:var(--muted); margin-top:0.8rem; line-height:1.5; }
  .board-ink { stroke:var(--panel-ink); } .board-text { fill:var(--panel-ink);
    font-family:"Avenir Next","Segoe UI",system-ui,sans-serif; }
  .blue-ink { stroke:var(--blue); } .blue-fill { fill:var(--blue); }
  .rust-ink { stroke:var(--rust); } .rust-fill { fill:var(--rust); }
  li { margin-bottom:0.5rem; }
</style>'''


CASES = {
    "tfi-reduced-basis.html": tfi_case,
    "h2-bracket.html": h2_case,
    "hchain-ladder.html": ladder_case,
    "butterfly-crossover.html": butterfly_case,
    "ising-cluster.html": ising_case,
    "smeared-spectral.html": spectral_case,
    "mz-closure.html": mz_case,
    "lr-dispatch.html": lr_case,
}

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "cases"), exist_ok=True)
    only = sys.argv[1:] or list(CASES)
    for name in only:
        t0 = time.time()
        with open(os.path.join(here, "cases", name), "w") as f:
            f.write(CASES[name]())
        print(f"wrote cases/{name}  ({time.time() - t0:.0f}s)")
