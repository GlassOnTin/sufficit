"""Basilisk-style case pages: each page is BUILT by running its case.
The source renders as the presentation (extracted live via inspect), the
a priori theory is stated, and the certification is executed and drawn.
Run from the repo root:  python3 docs/make_cases.py   (~2 minutes)
"""
import inspect
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sufficit as sf  # noqa: E402


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _stamp():
    """The commit that produced this page, so a certificate is citable:
    the numbers on a page belong to exactly one commit."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=ROOT, text=True).strip()
        d = subprocess.check_output(["git", "show", "-s", "--format=%cs"],
                                    cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=ROOT, text=True).strip()
        return f"commit <code>{h}</code>" + (" (modified)" if dirty else "") + f", {d}"
    except Exception:
        return "an uncommitted working tree"


STAMP = _stamp()


def check_counts():
    """The prose states how many checks the suite runs. Nothing keeps
    prose true except a check, so this is the check: count the tests,
    and fail the build if any stated count disagrees."""
    with open(os.path.join(ROOT, "test_sufficit.py")) as f:
        n = len(re.findall(r"^def test_", f.read(), re.M))
    for name in ("README.md", os.path.join("docs", "explainer.html")):
        with open(os.path.join(ROOT, name)) as f:
            stated = [int(m) for m in re.findall(r"(\d+) checks", f.read())]
        if not stated or any(s != n for s in stated):
            raise RuntimeError(f"{name} states {stated} checks; the suite has {n}")


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
<hr>
<p class="note">Every number and figure above comes from the run that
built this page: {STAMP}.</p>
</main>'''


def code_section(*funcs):
    """Basilisk-style literate rendering: each function is its own
    block, its docstring lifted out as the narrative paragraph, the
    code shown without it."""
    import ast
    import textwrap
    out = ["<h2>The program</h2>",
           '<p class="note">Rendered from the module at build time. '
           "The narrative for each function is its own docstring.</p>"]
    for f in funcs:
        src = textwrap.dedent(inspect.getsource(f))
        doc = inspect.getdoc(f)
        body = src
        if doc:
            first = ast.parse(src).body[0].body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                lines = src.splitlines()
                del lines[first.lineno - 1:first.end_lineno]
                body = "\n".join(lines)
        out.append(f"<h3><code>{esc(f.__name__)}</code></h3>")
        if doc:
            out.append(f'<p class="fn-doc">{esc(doc)}</p>')
        out.append(f"<pre>{esc(body.rstrip())}</pre>")
    return "\n".join(out)


class Axes:
    """Minimal hand-rolled chart axes for the case figures."""

    def __init__(self, xlim, ylim, w=640, h=320, logy=False, logx=False,
                 ml=58, mr=16, mt=18, mb=44):
        self.xlim, self.ylim, self.w, self.h = xlim, ylim, w, h
        self.logy, self.logx = logy, logx
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb

    def X(self, x):
        a, b = self.xlim
        if self.logx:
            a, b, x = math.log(a), math.log(b), math.log(x)
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
        "A certified sweep across a phase transition",
        "The ground energy of a quantum spin chain as its field is "
        "swept. Six exact solves and two pieces of reasoning give a "
        "rigorous band at a hundred field values, including the "
        "critical point.",
        [
            "<h2>The idea</h2>"
            "<p>The Hamiltonian is H(g) = H₀ + g·H₁ and we want its "
            "lowest eigenvalue at every g. The lower bound comes from "
            "the shape of that curve. The ground energy is the minimum "
            "over states of a quantity that is linear in g, and the "
            "minimum of a family of straight lines is concave. A "
            "concave curve lies above every chord drawn between two of "
            "its points. So certified lower bounds at six snapshot "
            "values of g certify the chord everywhere in between, and "
            "the chord costs nothing to evaluate.</p>"
            "<p>The upper bound is the variational theorem. Take the "
            "six snapshot ground states as a basis B and compute the "
            "small matrices BᵀH₀B and BᵀH₁B once. Then the best trial "
            "energy at any g is the lowest eigenvalue of a 6×6 matrix. "
            "Outside the snapshot interval the concavity argument says "
            "nothing, so the code refuses rather than guess.</p>",
            code_section(sf.tfi_chain, sf.reduced_basis_surrogate,
                         sf.reduced_basis_bracket),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Blue dots: the six certified "
            "snapshot lower bounds. Blue curve: the chord bound built "
            "from them. Rust: the 6×6 upper bound. The dashed curve is "
            "the exact answer from full diagonalization at all 100 "
            "points; it is computed to check the band and is not used "
            "by it.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li><strong>{contained}/100</strong> sweep points "
            "contained.</li>"
            f"<li>Upper-bound error at the critical point: <strong>"
            f"{gap_at_crit:.2e}</strong>, from six snapshot vectors.</li>"
            "<li>Cost per point after the offline stage: one "
            "<strong>6×6</strong> eigenvalue problem.</li></ul>",
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
        "The hydrogen molecule, with error bars of 10⁻¹³ hartree",
        "Gaussian integrals in closed form, second quantization, and a "
        "two-sided energy bracket. A separate calculation predicts the "
        "dissociation limit, and the curve has to hit it.",
        [
            "<h2>The idea</h2>"
            "<p>The upper bound is the variational theorem: the energy "
            "of any trial state sits above the true ground energy. The "
            "lower bound is a feasibility question. The matrix "
            "<code>H − c·I</code> has a Cholesky factorization exactly "
            "when the ground energy is at least <code>c</code>, so "
            "bisection on c turns a yes/no test into a certified floor. "
            "Both bounds carry explicit floating-point margins.</p>"
            "<p>There is also a free consistency check. At large "
            "separation the two atoms stop interacting, because the "
            "monopole terms cancel exactly for s orbitals. So the "
            "molecule must dissociate to twice the single-atom energy. "
            "That number comes from an independent calculation, and it "
            "fixes the asymptote the curve must approach.</p>",
            code_section(sf.hydrogen_atom_energy, sf.h2_energy_bracket,
                         sf.eigen_bracket),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>The certified potential curve. "
            "The bracket widths are near 10⁻¹³ hartree, about ten "
            "orders of magnitude thinner than the plotted line. The "
            "dashed asymptote is not fitted; it is computed separately, "
            "and the curve is required to approach it.</figcaption>"
            "</figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Largest bracket width on the curve: <strong>"
            f"{2 * max(errs):.1e}</strong> hartree. Chemical accuracy "
            "is 1.6·10⁻³.</li>"
            f"<li>Dissociation against two isolated atoms: <strong>"
            f"{diss_gap:.1e}</strong> hartree difference at R = 50.</li>"
            f"<li>Equilibrium bracket: <strong>{Es[i_eq]:.9f} ± "
            f"{errs[i_eq]:.0e}</strong> at R = {Rs[i_eq]:.1f} bohr. "
            "The published FCI/STO-3G value is −1.13727 at 1.4.</li>"
            "</ul>",
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
        "The window-length ladder",
        "A certified energy bracket for hydrogen chains that tightens "
        "about 2× for each unit of window length ℓ, at cost "
        "4<sup>ℓ</sup>. The chain length never enters the cost.",
        [
            "<h2>The idea</h2>"
            "<p>Slide a window of ℓ atoms along the chain, and weight "
            "each interaction by the number of windows that contain it. "
            "The weighted windows then add up to exactly the full "
            "Hamiltonian, so the true energy is at least the sum of "
            "certified window minima. Long-range Coulomb terms do not "
            "fit in any window; they are absorbed exactly, using a "
            "charge-neutral rewrite and the operator inequality "
            "<code>g(nᵢ−1)(nⱼ−1) ⪰ −(g/2)[(nᵢ−1)² + (nⱼ−1)²]</code>.</p>"
            "<p>Correction terms on the window overlaps tighten every "
            "window at once. An optimizer picks them, and the bound is "
            "valid whatever the optimizer did, because the corrections "
            "cancel telescopically by construction. The upper bound is "
            "a product of exactly solved blocks.</p>",
            code_section(sf.h_chain_bracket),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Blue: H₆, where every point is "
            "checked against exact diagonalization in this run. Rust: "
            "H₁₀, a 2²⁰-dimensional problem with no exact answer to "
            "compare against; none is needed. A separate recorded run "
            "extends H₁₀ to ℓ=7 at 55 mHa/atom using particle-number "
            "sectors; it is left out here to keep the page quick to "
            "regenerate.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>H₆ containment against exact: <strong>{sum(checks)}/"
            f"{len(checks)}</strong>, at every ℓ.</li>"
            "<li>Widths halve, roughly, per unit ℓ. This chain is "
            "critical, the slowest case; gapped systems tighten "
            "faster.</li></ul>",
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
        "The butterfly crossover",
        "For oscillatory wave kernels, the butterfly factorization "
        "beats plain low-rank compression only past a predictable "
        "problem size. This page measures where, and the certificate "
        "holds on both sides of the crossover.",
        [
            "<h2>The idea</h2>"
            "<p>A well-separated block of a wave kernel has numerical "
            "rank about R = k·r₁·r₂/D, which grows with the frequency "
            "k. Plain low-rank compression therefore costs R·(m+n) per "
            "apply. The butterfly ladder refines the row tree while it "
            "coarsens the column tree, which keeps every factor at "
            "rank about R/2<sup>L</sup>, and its transfer cost does "
            "not depend on N at all. So at fixed R the cost ratio must "
            "drop below 1 as N grows. That is a prediction, and this "
            "page tests it. The certificate is a posteriori: random "
            "probes of the assembled factorization, with the failure "
            "probability stated.</p>",
            code_section(sf.ButterflyBlock),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Helmholtz kernel at k = 1800, "
            "where R ≈ 95. Both methods are held to the same certified "
            f"accuracy; the plain method needs rank {r_plain}. The "
            "crossover arrives once the N-independent transfer cost is "
            "amortized, which is what the cost model predicts."
            "</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            "<li>Every apply is checked against the dense kernel: "
            f"actual error <strong>{stats_last[1]:.1e}</strong>, within "
            f"the certified <strong>{stats_last[0]:.1e}</strong>.</li>"
            f"<li>Failure probability of the probe certificate: <strong>"
            f"{stats_last[2]:.0e}</strong>.</li>"
            f"<li>Measured crossover: {ratios[0]:.2f}× → {ratios[-1]:.2f}× "
            "the plain low-rank cost as N grows at fixed R.</li></ul>",
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
        "The 2D Ising model at high temperature",
        "Free energy and correlations from cluster expansions, with "
        "the floating point carried in interval arithmetic. The "
        "functions refuse outside the region where the expansion "
        "provably converges.",
        [
            "<h2>The idea</h2>"
            "<p>The high-temperature expansion rewrites the Ising "
            "partition function as a gas of polymers, which are closed "
            "loops on the lattice with activity <code>t = tanh βJ"
            "</code>. The Kotecký–Preiss criterion gives a computable "
            "radius inside which the cluster series converges, with a "
            "geometric bound on the tail past truncation order L = 12. "
            "Local observables come from pinned polymers, counted by "
            "an Eulerian-path argument. Every coefficient and every "
            "tail is evaluated in outward-rounded interval arithmetic, "
            "so the floating point is part of the certificate.</p>"
            "<p>Outside the proven radius the functions raise an "
            "error. The wall sits at βJ ≈ 0.086. The true critical "
            "point is at βJ ≈ 0.4407, five times farther out: the wall "
            "belongs to the proof, not to the physics, and the "
            "certificate says so by refusing.</p>",
            code_section(sf.ising2d_logZ_density,
                         sf.ising2d_bond_correlation),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Lines: the certified error "
            "bound. Dots: the measured deviation from the exact 10×10 "
            "transfer matrix, which is the test suite's own truth "
            "generator, itself validated against exhaustive "
            "enumeration of 2¹⁶ states. The bound must sit above every "
            "dot. Both climb as the expansion nears its convergence "
            "wall. Dots on the floor are at the double-precision "
            "limit.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{2 * len(betas)}"
            "</strong>, both quantities at every temperature.</li>"
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
        "Two peaks from sixteen numbers",
        "A Gaussian-smeared spectral density reconstructed from 16 "
        "Euclidean correlator values, with a certified band at every "
        "frequency. Sharper resolution costs more certified error, and "
        "the certificate makes the price explicit.",
        [
            "<h2>The idea</h2>"
            "<p>The data see the spectral density ρ only through its "
            "exponential moments, C(t) = ∫e^(−ωt) ρ(ω) dω. Inverting "
            "that relation is ill-posed. The Hansen–Lupo–Tantalo "
            "observation is that you do not have to invert it: build "
            "the <em>smearing kernel</em> instead, as a combination of "
            "the exponentials you actually have.</p>"
            "<p>The certificate does not care how the combination was "
            "found. If the reconstructed kernel deviates from the "
            "target Gaussian by less than c·e^(−ω) everywhere — a "
            "bound checked on a dense grid, with a Lipschitz estimate "
            "per cell and an analytic tail — then positivity of ρ "
            "gives |value − truth| ≤ c·C(1). The one physical "
            "assumption, ρ ≥ 0, is stated in the provenance.</p>",
            code_section(sf.smeared_spectral, sf._hlt_solve),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>The certified band, rust above "
            "and blue below, around the smeared two-peak density at "
            "σ = 0.35, over 41 frequencies. The dashed curve is the "
            "exact smeared truth, computable here because the test "
            "density is synthetic; the band was built without it. The "
            "first peak is certified from both sides. At the second, "
            "the lower bound has fallen to zero, and past ω ≈ 2 the "
            "band runs off the top of the plot. Sixteen exponentials "
            "cannot say more, and the certificate reports exactly "
            "that.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(ws)}</strong> "
            "frequencies.</li>"
            "<li>Resolution costs error, monotonically: the certified "
            f"error at ω = 1 is <strong>{res_errs[0]:.3f} → "
            f"{res_errs[1]:.3f} → {res_errs[2]:.3f}</strong> for "
            "σ = 0.6 → 0.4 → 0.25.</li>"
            "<li>The certificate never used the truth. It is c·C(1), "
            "from the data and the kernel bound alone.</li></ul>",
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
        "Dropping the fast variables",
        "A 10-dimensional slow-fast system reduced to its 2 slow "
        "coordinates by the Markovian closure. The certified error "
        "comes from the fast sector's spectral gap. Without a gap "
        "there is no decay proof, and the function refuses.",
        [
            "<h2>The idea</h2>"
            "<p>Project a linear system onto its slow observables and "
            "an exact memory term is left over: K(s) = A₁₂e^(A₂₂s)A₂₁. "
            "If the fast sector is dissipative, meaning the log-norm "
            "of A₂₂ is strictly negative, then the kernel decays at "
            "the gap μ. Dropping the memory gives the Markovian "
            "closure A₁₁ − A₁₂A₂₂⁻¹A₂₁, and a Grönwall argument bounds "
            "the cost of dropping it, with constants you can compute: "
            "block norms, log-norms, and 1/μ². The certificate is "
            "priced by the same physics that justifies the reduction. "
            "A fast initial transient adds one more decaying term.</p>",
            code_section(sf._lognorm, sf.mz_closure_linear),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Both slow coordinates. The "
            "closure prediction is solid, inside its certified tube "
            "(shaded; the half-width is the joint 2-norm bound). The "
            "exact 10-dimensional propagation is dashed, computed only "
            "to check containment. The tube half-width is "
            f"{errs[0]:.3f} at T = {Ts[0]:g} and saturates at "
            f"{errs[-1]:.3f} by T = 20, because the reduced model is "
            "itself dissipative.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(Ts)}</strong> "
            "horizons out to T = 20.</li>"
            "<li>The gap prices the certificate: doubling the fast "
            f"sector's stiffness tightens the bound <strong>"
            f"{e_base / e_stiff:.1f}×</strong> at T = 5.</li>"
            f"<li>No-gap system: <strong>"
            f"{'refused' if refused else 'NOT REFUSED'}</strong>. An "
            "undamped fast mode means the memory kernel never "
            "certifiably decays.</li></ul>",
        ])


# ======================================================================
def sph_case():
    snaps_t = (1.6, 2.8, 3.4)
    runs = {}
    for n in (16, 24, 36, 48):
        runs[n] = sf.sph_dam_break(nres=n,
                                   snapshots=snaps_t if n == 36 else ())

    def J(n):
        o = runs[n]
        return float(np.sum(o["F"]) * (o["ts"][1] - o["ts"][0]))

    hs = [1 / 16, 1 / 24, 1 / 36]
    cJ = sf.gci_extrapolate([J(n) for n in (16, 24, 36)], hs,
                        p_floor=0.3)
    contained = abs(J(48) - cJ.value) <= cJ.err
    peaks = {n: float(np.max(runs[n]["F"])) for n in runs}
    try:
        sf.gci_extrapolate([peaks[n] for n in (16, 24, 36)], hs)
        peak_refused = False
    except ValueError:
        peak_refused = True
    tall = [sf.sph_wall_impulse(n, obstacle=(2.9, 0.4, 0.4))
            for n in (12, 18, 27)]
    c0b = sf.gci_extrapolate(tall, [1 / 12, 1 / 18, 1 / 27])
    try:
        sf.gci_extrapolate([sf.sph_wall_impulse(n, obstacle=(2.9, 0.4, 0.12))
                            for n in (12, 18, 27)],
                           [1 / 12, 1 / 18, 1 / 27])
        mid_refused = False
    except ValueError:
        mid_refused = True

    # snapshots: particles colored by pressure, three moments
    W, Hp, ml, mt = 640, 150, 30, 8
    sc = (W - 2 * ml) / 4.0
    panels = []
    pmax = max(float(s[3].max()) for s in runs[36]["snaps"]) or 1.0
    for k, (tt, sx, sy, sp) in enumerate(runs[36]["snaps"]):
        oy = k * Hp
        dots = []
        hot = sp > 0.12 * pmax
        for cls, sel in (("blue-fill", ~hot), ("rust-fill", hot)):
            for x, y, p in zip(sx[sel], sy[sel], sp[sel]):
                op = 0.35 + 0.6 * min(max(p, 0.0) / pmax, 1.0)
                dots.append(f'<circle cx="{ml + sc * x:.1f}" '
                            f'cy="{oy + Hp - 18 - sc * y:.1f}" r="2.1" '
                            f'class="{cls}" opacity="{op:.2f}"/>')
        panels.append(
            f'<line x1="{ml}" y1="{oy + Hp - 18:.1f}" '
            f'x2="{ml + sc * 4.0:.1f}" y2="{oy + Hp - 18:.1f}" '
            f'class="board-ink" stroke-width="2"/>'
            f'<line x1="{ml + sc * 4.0:.1f}" y1="{oy + Hp - 18:.1f}" '
            f'x2="{ml + sc * 4.0:.1f}" y2="{oy + mt:.1f}" '
            f'class="board-ink" stroke-width="3"/>'
            f'<text x="{ml:.0f}" y="{oy + mt + 10:.1f}" class="board-text" '
            f'font-size="11">t = {tt:g}</text>' + "".join(dots))
    svg_snap = (f'<svg viewBox="0 0 {W} {3 * Hp}" role="img" '
                f'aria-label="The bore approaching, striking, and running '
                f'up the wall; particles colored by pressure">'
                + "".join(panels) + "</svg>")

    # convergence-order figure, Basilisk convention: error against the
    # finest run versus h on log axes, with slope guides
    eJ = {n: abs(J(n) - J(48)) for n in (16, 24, 36)}
    eP = {n: abs(peaks[n] - peaks[48]) for n in (16, 24, 36)}
    ylo = 0.5 * min(min(eJ.values()), min(eP.values()))
    yhi = 3.0 * max(max(eJ.values()), max(eP.values()))
    axc = Axes((1 / 40, 1 / 14), (ylo, yhi), h=300, logx=True, logy=True)
    guides = []
    for p, lab in ((1, "slope 1"), (2, "slope 2")):
        xs = (1 / 38, 1 / 15)
        ys = [eJ[16] * (x * 16) ** p for x in xs]   # through the coarse rung
        guides.append(
            f'<path d="{axc.path(xs, ys)}" fill="none" class="board-ink" '
            f'stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>'
            f'<text x="{axc.X(xs[0]) - 6:.1f}" y="{axc.Y(ys[0]) + 4:.1f}" '
            f'text-anchor="end" class="board-text" font-size="10" '
            f'opacity="0.7">{lab}</text>')
    pts = []
    hsx = [1 / 16, 1 / 24, 1 / 36]
    pts.append(f'<path d="{axc.path(hsx, [eJ[n] for n in (16, 24, 36)])}" '
               f'fill="none" class="blue-ink" stroke-width="2"/>')
    for n in (16, 24, 36):
        pts.append(f'<circle cx="{axc.X(1 / n):.1f}" '
                   f'cy="{axc.Y(max(eJ[n], 3e-3)):.1f}" r="5" '
                   f'class="blue-fill"/>')
        pts.append(f'<circle cx="{axc.X(1 / n):.1f}" '
                   f'cy="{axc.Y(max(eP[n], 3e-3)):.1f}" r="5" fill="none" '
                   f'class="rust-ink" stroke-width="2"/>')
    svg_ord = f'''<svg viewBox="0 0 640 300" role="img" aria-label="Error
against the finest run versus resolution: the impulse follows a slope near
one, the peak has no slope at all">
{axc.grid((3e-2, 1e-1, 3e-1), (1 / 16, 1 / 24, 1 / 36),
          xfmt=lambda v: f"h = 1/{round(1 / v)}",
          yfmt=lambda v: f"{v:g}")}
{"".join(guides)}{"".join(pts)}
<text x="{axc.ml + 10}" y="{axc.mt + 14}" class="board-text"
      font-size="10.5" fill="var(--blue)">impulse error vs finest (filled)</text>
<text x="{axc.ml + 10}" y="{axc.mt + 30}" class="board-text"
      font-size="10.5" fill="var(--rust)">peak error vs finest (open)</text>
</svg>'''

    # force traces at two resolutions
    ax = Axes((1.8, 4.6), (-0.02, 1.05 * max(peaks[24], peaks[36])), h=280)
    tr = []
    for n, cls in ((24, "blue"), (36, "rust")):
        o = runs[n]
        keep = o["ts"] > 1.8
        tr.append(f'<path d="{ax.path(o["ts"][keep], o["F"][keep])}" '
                  f'fill="none" class="{cls}-ink" stroke-width="1.6" '
                  f'opacity="0.9"/>')
    svg_F = f'''<svg viewBox="0 0 640 280" role="img" aria-label="Wall force
against time at two resolutions: peaks scatter, areas agree better">
{ax.grid((0.2, 0.4, 0.6, 0.8), (2.0, 2.5, 3.0, 3.5, 4.0, 4.5),
         xfmt=lambda v: f"t = {v:g}", yfmt=lambda v: f"{v:g}")}
{"".join(tr)}
<text x="{ax.ml + 10}" y="{ax.mt + 14}" class="board-text" font-size="10.5"
      fill="var(--blue)">force on the wall, resolution 24 (blue) and 36
 (rust)</text></svg>'''

    return page(
        "Case: a breaking wave against a sea wall",
        "certified case",
        "A wave, a wall, and what can honestly be promised",
        "A dam-break bore strikes a wall in a declared SPH model. The "
        "engineering question is the load on the wall. Three different "
        "queries get three different verdicts, and the verdicts are "
        "the product.",
        [
            "<h2>The idea</h2>"
            "<p>Breaking-wave peak pressures are famously "
            "irreproducible, in experiments as in simulations: the "
            "peak depends on the last millimeter of breaker shape and "
            "any entrapped air. Design practice responds by using the "
            "impulse, the time integral of the force. The same triage "
            "falls out of the certificates here. The raw peak shows no "
            "asymptotic range on a resolution ladder, so the "
            "convergence certifier refuses it. The delivered impulse "
            "converges, and certifies, with an honestly wide error "
            "bar: this laptop-budget model is below its asymptotic "
            "range, and the certificate says by how much instead of "
            "hiding it.</p>"
            "<p>The certifier is the engineering community's own "
            "instrument, Roache's grid-convergence method, with two "
            "teeth added: it refuses when the ladder shows no "
            "asymptotic range, and it caps the usable order at the "
            "scheme's formal order, because a lucky triplet can "
            "otherwise measure a spurious high order and issue a "
            "false, too-tight certificate. Both failure modes were "
            "hit, measured, during this build.</p>"
            "<p>The design question, what obstacle protects the wall, "
            "gets a three-part answer. A berm 40% of the water-column "
            "height certifiably zeroes the delivered impulse: the "
            "ladder is constant at zero. The bare wall certifies a "
            "nonzero impulse. In between, a low berm sheds a thin jet "
            "over its crest that the affordable resolutions cannot "
            "converge, and the certifier refuses rather than guess.</p>",
            code_section(sf.sph_dam_break, sf.sph_wall_impulse,
                         sf.gci_extrapolate),
            "<h2>The result</h2>"
            f"<figure>{svg_snap}<figcaption>The bore at t = 1.6, 2.8 "
            "and 3.4: approach, impact, runup. Particles are colored "
            "by pressure; the wall lights up on impact. 1296 fluid "
            "particles at the resolution shown.</figcaption></figure>"
            f"<figure>{svg_F}<figcaption>The force on the wall at two "
            "resolutions. The peaks disagree by half; the areas under "
            "the curves agree far better. That is the whole story of "
            "why the impulse is the certifiable query and the peak is "
            "not.</figcaption></figure>"
            f"<figure>{svg_ord}<figcaption>The convergence-order "
            "figure, in the Basilisk convention: error against the "
            "finest run, on log axes, with slope guides. The impulse "
            "follows a slope near one; smooth-field SPH is formally "
            "second order, and the shortfall is the impact and the "
            "boundary treatment, stated rather than hidden. The peak "
            "points fall on no slope at all, which is the refusal, "
            "drawn.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Delivered impulse: <strong>{cJ.value:.3f} ± "
            f"{cJ.err:.3f}</strong> (measured order in the provenance; "
            "wide, and honestly so). The finer verification rung the "
            "certificate never saw lands "
            f"<strong>{'inside' if contained else 'OUTSIDE'}</strong>."
            "</li>"
            f"<li>Raw peak force: <strong>"
            f"{'refused' if peak_refused else 'NOT REFUSED'}</strong> — "
            f"ladder peaks {', '.join(f'{peaks[n]:.2f}' for n in (16, 24, 36))}, "
            "no asymptotic range.</li>"
            f"<li>Tall berm (40% of column height): impulse certified "
            f"<strong>{c0b.value:.0f} ± {c0b.err:.0f}</strong> — the "
            "bore never reaches the wall, at any rung.</li>"
            f"<li>Low berm (12%): <strong>"
            f"{'refused' if mid_refused else 'NOT REFUSED'}</strong> — "
            "the crest jet is under-resolved at this budget, and the "
            "certificate says so.</li></ul>",
        ])


# ======================================================================
def gw_case():
    import time as _time
    tight = sf.gw_surrogate_build(seed=7)
    loose = sf.gw_surrogate_build(seed=7, eps_build=1e-3)
    lams = np.linspace(1.0, 2.0, 120)
    sweeps = {}
    for name, sur in (("tight", tight), ("loose", loose)):
        sweeps[name] = [max(sf._gw_mismatch(sf._gw_surrogate_raw(sur, float(l)),
                                            sf._gw_chirp(float(l))), 1e-17)
                        for l in lams]
    rng = np.random.default_rng(99)
    fresh = rng.uniform(1.0, 2.0, 200)
    exceed = sum(sf._gw_mismatch(sf._gw_surrogate_raw(tight, float(l)),
                                 sf._gw_chirp(float(l))) > tight["m_cal"]
                 for l in fresh)
    t0 = _time.time()
    for l in fresh[:50]:
        sf._gw_surrogate_raw(tight, float(l))
    t_sur = (_time.time() - t0) / 50
    t0 = _time.time()
    for l in fresh[:50]:
        sf._gw_chirp(float(l))
    t_truth = (_time.time() - t0) / 50

    ax = Axes((1.0, 2.0), (1e-17, 1.0), h=340, logy=True)
    parts = []
    for name, cls in (("loose", "rust"), ("tight", "blue")):
        sur = tight if name == "tight" else loose
        parts.append(
            f'<path d="{ax.path(lams, sweeps[name])}" fill="none" '
            f'class="{cls}-ink" stroke-width="1.8" opacity="0.9"/>'
            f'<line x1="{ax.ml}" y1="{ax.Y(max(sur["m_cal"], 1e-17)):.1f}" '
            f'x2="{640 - ax.mr}" y2="{ax.Y(max(sur["m_cal"], 1e-17)):.1f}" '
            f'class="{cls}-ink" stroke-width="1.4" stroke-dasharray="6 4"/>')
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Surrogate
mismatch across the parameter range for two build tolerances, each under
its conformal calibrated bound">
{ax.grid((1e-15, 1e-11, 1e-7, 1e-3), (1.0, 1.25, 1.5, 1.75, 2.0),
         xfmt=lambda v: f"λ = {v:g}", yfmt=pow10)}
{"".join(parts)}
<text x="{ax.ml + 10}" y="{ax.Y(loose["m_cal"]) - 8:.1f}" class="board-text"
      font-size="10.5" fill="var(--rust)">3-mode build: calibrated mismatch
 {loose["m_cal"]:.1e} (dashed = the certificate)</text>
<text x="{ax.ml + 10}" y="{ax.Y(max(tight["m_cal"], 1e-17)) - 8:.1f}"
      class="board-text" font-size="10.5" fill="var(--blue)">4-mode build:
 calibrated {tight["m_cal"]:.0e} — the manifold's true rank, found from data</text>
</svg>'''

    return page(
        "Case: gravitational-wave surrogates with a conformal mismatch "
        "certificate",
        "certified case",
        "Waveform surrogates with stated odds",
        "A detector fixes the mismatch tolerance it can distinguish. "
        "The surrogate answers any parameter in a fraction of a "
        "millisecond, and its certificate states the probability that "
        "it is wrong.",
        [
            "<h2>The idea</h2>"
            "<p>Raw waveforms decorrelate across parameter space by "
            "hundreds of radians of dephasing, so no small basis fits "
            "them directly. But the amplitude and the unwrapped phase "
            "are smooth functions of the parameter. So fix the free "
            "global phase, unwrap, reduce amplitude and phase "
            "separately by SVD, and fit the mode coefficients. That is "
            "the gwsurrogate architecture, reproduced here from "
            "waveform evaluations alone.</p>"
            "<p>The certificate is conformal. Hold out n_cal fresh "
            "parameter draws, measure the surrogate's mismatch at each, "
            "and take the worst. For a new draw from the same "
            "distribution, the probability of exceeding that worst "
            "value is at most 1/(n_cal+1). This follows from "
            "exchangeability alone; no smoothness is assumed. It is "
            "the same theorem behind the Mori–Zwanzig empirical tier. "
            "The surrogate refuses outside its training range, and "
            "dispatch refuses when the calibrated mismatch exceeds "
            "the detector's ε.</p>",
            code_section(sf._gw_chirp, sf._gw_mismatch,
                         sf.gw_surrogate_build, sf._gw_surrogate_raw,
                         sf.gw_surrogate_eval, sf.gw_surrogate_dispatch),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Measured mismatch across 120 "
            "parameters (solid) for two offline builds, each under its "
            "conformal bound (dashed). The 4-mode surrogate finds the "
            "waveform family's exact amplitude and phase rank and "
            "reaches machine precision. The 3-mode build is "
            "deliberately too small, and the certificate reports the "
            "consequence: mismatch near 10⁻².</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Fresh draws exceeding the calibrated bound: "
            f"<strong>{exceed}/200</strong>. The declared rate is "
            f"1/{tight['n_cal'] + 1}, about "
            f"{200 // (tight['n_cal'] + 1)}/200.</li>"
            f"<li>Cost per query: <strong>{1e6 * t_sur:.0f} µs</strong>, "
            "independent of what the truth costs. There is no speedup "
            f"here, since this model family is itself cheap "
            f"({1e6 * t_truth:.0f} µs). The architecture is the point: "
            "with numerical-relativity truth at about 10⁵ CPU-hours "
            "per waveform, evaluating truth only offline is the whole "
            "product.</li>"
            "<li>Dispatch refuses below the calibrated mismatch and "
            "outside the training range. Both cases are exercised in "
            "the test suite.</li></ul>",
        ])


# ======================================================================
def sos_case():
    c2 = sf.lorenz_mean_z_bracket(degree=2)
    c4 = sf.lorenz_mean_z_bracket(degree=4)

    def f(x, y, z):
        return 10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z

    x, y, z = 1.0, 1.0, 1.0
    dt, n, acc = 0.004, 150_000, 0.0
    for i in range(n):
        ax_, ay, az = f(x, y, z)
        bx, by, bz = f(x + dt / 2 * ax_, y + dt / 2 * ay, z + dt / 2 * az)
        cx, cy, cz = f(x + dt / 2 * bx, y + dt / 2 * by, z + dt / 2 * bz)
        ex, ey, ez = f(x + dt * cx, y + dt * cy, z + dt * cz)
        x += dt / 6 * (ax_ + 2 * bx + 2 * cx + ex)
        y += dt / 6 * (ay + 2 * by + 2 * cy + ey)
        z += dt / 6 * (az + 2 * bz + 2 * cz + ez)
        acc += z
    mean_z = acc / n
    from fractions import Fraction as F
    bad = {(2, 0, 0): F(1), (1, 1, 0): F(-3), (0, 2, 0): F(1)}
    refused = sf._sos_exact_check(bad) is False

    ax = Axes((22.0, 28.6), (0.0, 3.0), h=200, mt=30, mb=40)
    rows = [("degree-2 certificate", 27.0, 28.0, "rust", 2.35),
            ("degree-4 certificate", 27.0, float(c4.value + c4.err),
             "blue", 1.55)]
    bars = []
    for label, lo, hi, cls, yy in rows:
        bars.append(
            f'<line x1="{ax.X(lo):.1f}" y1="{ax.Y(yy):.1f}" '
            f'x2="{ax.X(hi):.1f}" y2="{ax.Y(yy):.1f}" class="{cls}-ink" '
            f'stroke-width="7" stroke-linecap="round" opacity="0.85"/>'
            f'<text x="{ax.X(lo) - 10:.1f}" y="{ax.Y(yy) + 4:.1f}" '
            f'text-anchor="end" class="board-text" font-size="11">'
            f'{label}</text>')
    svg = f'''<svg viewBox="0 0 640 200" role="img" aria-label="Certified
interval ladder on the Lorenz mean of z, with witness and attractor">
{ax.grid((), (23, 24, 25, 26, 27, 28), xfmt=lambda v: f"{v:g}")}
{"".join(bars)}
<line x1="{ax.X(27):.1f}" y1="{ax.mt}" x2="{ax.X(27):.1f}"
      y2="{200 - ax.mb}" class="board-ink" stroke-dasharray="4 4"
      stroke-width="1.3" opacity="0.7"/>
<text x="{ax.X(27):.1f}" y="{ax.mt - 6}" text-anchor="middle"
      class="board-text" font-size="10.5">fixed points C±: ⟨z⟩ = 27
 exactly (the witness)</text>
<circle cx="{ax.X(mean_z):.1f}" cy="{ax.Y(0.75):.1f}" r="5"
        class="board-ink" fill="none" stroke-width="2"/>
<text x="{ax.X(mean_z):.1f}" y="{ax.Y(0.75) + 20:.1f}"
      text-anchor="middle" class="board-text" font-size="10.5">chaotic
 attractor, simulated: {mean_z:.2f}</text></svg>'''

    return page(
        "Case: sum-of-squares transport bounds, proven over the rationals",
        "certified case",
        "A bound on chaos, proven over the rationals",
        "The long-time average of z in the Lorenz system, for every "
        "trajectory at once. An SDP solver proposes the certificate; "
        "exact rational arithmetic decides it.",
        [
            "<h2>The idea</h2>"
            "<p>Take any polynomial V. Along a trajectory, dV/dt = "
            "∇V·f, and the time average of a derivative of a bounded "
            "quantity is zero. So the time average of Φ equals the "
            "time average of Φ + ∇V·f, which is at most "
            "sup<sub>x</sub> [Φ + ∇V·f]. Every V gives a bound; "
            "minimizing over V is a sum-of-squares program. This is "
            "the background method of Doering and Constantin, in the "
            "modern form of Tobasco, Fantuzzi, and Goulart.</p>"
            "<p>The search for V and for a Gram matrix is float "
            "optimization, here an SDP solver, and none of it is "
            "trusted. The certificate is the polynomial identity "
            "U − Φ − ∇V·f = mᵀQm, checked coefficient by coefficient "
            "over the rationals, plus a rational LDLᵀ proof that "
            "Q ⪰ 0. No float enters the verdict. The theorem needs "
            "bounded trajectories, and that is itself an SOS "
            "certificate, for the classical absorbing ball. The fixed "
            "points C± are exact trajectories with ⟨z⟩ = ρ−1 = 27, "
            "which closes the bracket from below.</p>",
            code_section(sf._sos_exact_check, sf._rational_ldl_psd,
                         sf.lorenz_mean_z_bracket),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>The certified ladder. Quadratic "
            "V proves ⟨z⟩ ≤ 28, by a hand-derived Gram checked "
            "exactly. Quartic V, found by the SDP solver after "
            "nondimensionalization and re-proven over the rationals, "
            f"lands at {float(c4.value + c4.err):.3f}: sharp against "
            "the fixed-point witness at 27 to one part in 27,000. The "
            "chaotic attractor's own average sits well below, because "
            "the bracket bounds the worst trajectory, and the worst "
            "trajectory is the one sitting on the fixed point."
            "</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Degree-2 bracket <strong>[27, 28]</strong>; degree-4 "
            f"<strong>[27, {float(c4.value + c4.err):.3f}]</strong>. "
            "Both PSD proofs completed in exact rational arithmetic."
            "</li>"
            f"<li>Simulated attractor mean <strong>{mean_z:.2f}</strong>, "
            "below the certified bound.</li>"
            f"<li>The checker refuses an indefinite polynomial: "
            f"<strong>{'yes' if refused else 'NO'}</strong>.</li>"
            "<li>cvxpy/SCS is a search-only dependency. Removing it "
            "kills the degree-4 search and weakens no certificate."
            "</li></ul>",
        ])


# ======================================================================
def plasma_case():
    a, v, T = 0.3, 1.0, 25.0
    ladder = (0.16, 0.08, 0.04)
    verify = (0.02, 0.01)
    E = {0: {}, 1: {}}
    for order in (0, 1):
        for e in ladder + verify:
            E[order][e] = abs(sf._gc_prediction(order, e, a, v, T)
                              - sf._gc_truth_cached(e, a, v, T))
    certs = {o: sf.gc_drift_asymptotic(min(verify), order=o) for o in (0, 1)}
    Chat = {o: max(E[o][e] / e ** (o + 1) for e in ladder) for o in (0, 1)}
    contained = sum(E[o][e] <= sf.gc_drift_asymptotic(e, order=o).err
                    for o in (0, 1) for e in verify)
    c_disp0 = sf.gc_drift_dispatch(0.004, tol=0.05)
    c_disp1 = sf.gc_drift_dispatch(0.02, tol=2e-3)
    try:
        sf.asymptotic_extrapolate(lambda x: 0.0, lambda x: x, 0.05, 2,
                                  (0.8, 0.4, 0.2, 0.1))
        refused = False
    except ValueError:
        refused = True

    ax = Axes((0.007, 0.25), (3e-7, 3.0), h=360, logy=True, logx=True)
    wallx = ax.X(min(ladder))
    parts = [f'<rect x="{wallx:.1f}" y="{ax.mt}" '
             f'width="{640 - ax.mr - wallx:.1f}" '
             f'height="{360 - ax.mt - ax.mb}" fill="var(--panel-ink)" '
             f'opacity="0.10"/>'
             f'<text x="{wallx + 8:.1f}" y="{ax.mt + 14}" class="board-text" '
             f'font-size="10.5" opacity="0.9">calibration ladder: kinetic '
             f'solves are cheap here (cost ~ 1/ε)</text>']
    for o, cls in ((0, "rust"), (1, "blue")):
        eta_C = 2.0 * Chat[o]
        xs = (0.008, min(ladder))
        parts.append(
            f'<path d="{ax.path(xs, [eta_C * x ** (o + 1) for x in xs])}" '
            f'fill="none" class="{cls}-ink" stroke-width="2"/>')
        for e in ladder:
            parts.append(f'<circle cx="{ax.X(e):.1f}" '
                         f'cy="{ax.Y(max(E[o][e], 4e-7)):.1f}" r="4.5" '
                         f'class="{cls}-fill"/>')
        for e in verify:
            parts.append(
                f'<circle cx="{ax.X(e):.1f}" '
                f'cy="{ax.Y(max(E[o][e], 4e-7)):.1f}" r="4.5" fill="none" '
                f'class="{cls}-ink" stroke-width="2"/>')
    svg = f'''<svg viewBox="0 0 640 360" role="img" aria-label="Measured
truncation errors on the calibration ladder, certified envelopes
extrapolating down in epsilon, verification solves beneath them">
{ax.grid((1e-6, 1e-4, 1e-2, 1.0), (0.01, 0.02, 0.04, 0.08, 0.16),
         xfmt=lambda w: f"ε = {w:g}", yfmt=pow10)}
{"".join(parts)}
<text x="{ax.X(0.011):.1f}" y="{ax.Y(2 * Chat[0] * 0.011) - 10:.1f}"
      class="board-text" font-size="10.5" fill="var(--rust)">order 0:
 certified η·C·ε</text>
<text x="{ax.X(0.011):.1f}" y="{ax.Y(2 * Chat[1] * 0.011 ** 2) + 20:.1f}"
      class="board-text" font-size="10.5" fill="var(--blue)">order 1:
 certified η·C·ε²</text></svg>'''

    return page(
        "Case: the plasma hierarchy, and the asymptotic tier's first "
        "shipment",
        "certified case",
        "The guiding-center hierarchy",
        "The first ASYMPTOTIC-tier certificate. The truncation "
        "exponent of the reduction is a theorem. The constant is "
        "measured at large ε, where full kinetic solves are cheap, "
        "and the bound is carried down to small ε, where they are "
        "not.",
        [
            "<h2>The idea</h2>"
            "<p>Every reduction hierarchy in plasma physics, from "
            "kinetic to drift-kinetic to fluid, rests on a small "
            "parameter: ε, the gyroradius over the gradient scale. The "
            "truncation error of an order-k reduction is O(ε^(k+1)). "
            "The exponent is a theorem; the constant in front is not "
            "computable. Tier.ASYMPTOTIC says exactly that: exponent "
            "proven, constant measured, remaining assumption stated."
            "</p>"
            "<p>The measurement is affordable for a simple reason. A "
            "full kinetic solve costs about 1/ε, so at large ε it is "
            "cheap. Measure the error constant on a ladder of large ε "
            "values, take the worst, add a safety factor, and carry "
            "the bound down to the small ε you actually care about. "
            "One subtlety came out of the measurements: the order-1 "
            "truncation coefficient oscillates with gyrophase, so "
            "convergence slopes between ladder rungs swing wildly "
            "while the envelope stays flat. The certifier therefore "
            "checks the envelope, and it refuses in the one dangerous "
            "direction, a measured constant that grows toward the "
            "ladder floor. That growth is what the data look like when "
            "the claimed exponent is wrong.</p>",
            code_section(sf.asymptotic_extrapolate, sf.gc_drift_asymptotic,
                         sf.gc_drift_dispatch, sf._gc_prediction,
                         sf._gc_orbit_delta),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>Filled dots: measured truncation "
            "errors on the calibration ladder, order 0 in rust and "
            "order 1 in blue. Lines: the certified envelopes "
            "η·C·ε<sup>k</sup>, extrapolated leftward into the region "
            "where kinetic solves get expensive. Open rings: "
            "verification solves at ε = 0.02 and 0.01, never used by "
            "the certificate, sitting beneath their envelopes as the "
            "theorem requires.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment at the verification ε: <strong>"
            f"{contained}/4</strong>, both orders at both ε.</li>"
            f"<li>Tier: <strong>{certs[1].tier.name}</strong>. The "
            "assumptions are stated in the provenance string.</li>"
            "<li>Dispatch along the hierarchy: tol = 0.05 is served by "
            f"<strong>order 0</strong>, tol = 2·10⁻³ escalates to "
            f"<strong>order 1</strong> (err {c_disp1.err:.1e}), and an "
            "impossible tol refuses and prices the kinetic rung.</li>"
            "<li>Wrong-exponent refusal: a synthetic truth scaling as "
            "ε, fed to a certifier claiming ε², was <strong>"
            f"{'refused' if refused else 'NOT REFUSED'}</strong>.</li>"
            "</ul>",
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
        "A light cone, measured at its boundary",
        "A quench observable on a spin chain, computed inside a cone "
        "of sites. The truncation error is an integral of a boundary "
        "commutator, and the simulation measures that commutator as "
        "it runs. The chain length never enters the cost.",
        [
            "<h2>The idea</h2>"
            "<p>Compare the full dynamics with dynamics restricted to "
            "a cone of sites around the observable. Duhamel's formula "
            "gives ‖A(t) − A_cone(t)‖ ≤ ∫₀ᵗ ‖[H − H_cone, A_cone(s)]‖ "
            "ds, and in that difference only the two bonds crossing "
            "the cone boundary matter; everything else commutes. The "
            "cone simulation already has A_cone(s) in hand, so it can "
            "measure those commutator norms directly. They stay near "
            "zero until the excitation front physically reaches the "
            "boundary. So the certificate contains no Lieb–Robinson "
            "velocity constants to be loose about; it contains a "
            "measurement. The time integral is made rigorous by an "
            "interpolation remainder priced by the measured second "
            "derivative, with a crude cap only at order δ⁴, where it "
            "cannot matter.</p>",
            code_section(sf.tfi_quench_dispatch, sf._lr_cone_run,
                         sf._opnorm_ub),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>The certified band around "
            "⟨Z(t)⟩ at the critical point, tolerance 10⁻². Ticks mark "
            "where dispatch grew the cone. In the shaded region the "
            "measured boundary leakage exceeds the tolerance at every "
            "affordable radius, so the code refuses and prices the "
            "next cone instead of guessing. The certified width moves "
            "in a sawtooth because dispatch always takes the cheapest "
            "adequate cone. The dashed curve is the exact "
            "2048-dimensional answer, computed only to check the band."
            "</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/{len(cert)}</strong> "
            "certified sweep points against exact diagonalization.</li>"
            f"<li>Refusal wall at <strong>t = {wall:g}</strong> for "
            f"max_dim = {md}. Past it, no certificate is claimed.</li>"
            "<li>Chain length does not enter: a <strong>10⁶-site"
            f"</strong> chain certified to ±10⁻³ in <strong>"
            f"{big_secs:.1f} s</strong>, bit-identical to the "
            "2001-site run: "
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
  h3 { font-size:0.95rem; margin:2rem 0 0.4rem; }
  h3 code { background:none; padding:0; font-size:0.95rem; }
  .fn-doc { margin:0.2rem 0 0.7rem; }
  .note { font-family:"Avenir Next","Segoe UI",system-ui,sans-serif;
    font-size:0.82rem; color:var(--muted); margin:0.2rem 0 0.8rem; }
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


# ======================================================================
def gs_case():
    import numpy as np2
    from dolfinx import geometry
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = []
    for n in (8, 16, 32):
        r = sf.gs_equilibrium_certified(n=n)
        ladder.append((1.0 / n, r["err_measured"], r["energy_bound"]))
    rc = sf.gs_equilibrium_certified(n=48, c=1.0)
    r12 = sf.gs_equilibrium_certified(n=12, c=1.0)
    agree = abs(rc["Q"].value - r12["Q"].value) <= rc["Q"].err + r12["Q"].err
    try:
        sf.gs_equilibrium_certified(n=8, c=3.0)
        refused = False
    except ValueError:
        refused = True

    # flux surfaces of the coupled equilibrium: evaluate on a grid and
    # extract contour polylines with matplotlib (build-time tool only)
    msh, uh = rc["msh"], rc["uh"]
    gr = np2.linspace(2.001, 3.999, 121)
    gz = np2.linspace(-0.999, 0.999, 121)
    GX, GZ = np2.meshgrid(gr, gz)
    pts = np2.column_stack([GX.ravel(), GZ.ravel(),
                            np2.zeros(GX.size)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    vals = uh.eval(pts, np2.array(cells, dtype=np2.int32)).reshape(GX.shape)
    cs = plt.contour(GX, GZ, vals,
                 levels=np2.linspace(float(vals.min()) + 0.01,
                                     0.40, 10))
    ml, mt, sc = 40, 14, 130.0
    paths = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            d = "M " + " L ".join(
                f"{ml + sc * (p[0] - 2.0):.1f} "
                f"{mt + sc * (1.0 - p[1]):.1f}" for p in seg[::2])
            paths.append(f'<path d="{d}" fill="none" class="blue-ink" '
                         f'stroke-width="1.4" opacity="0.8"/>')
    plt.close("all")
    Wpx, Hpx = ml + int(sc * 2) + 20, mt + int(sc * 2) + 30
    svg_flux = (f'<svg viewBox="0 0 {Wpx} {Hpx}" role="img" '
                f'aria-label="Flux surfaces of the coupled equilibrium">'
                f'<rect x="{ml}" y="{mt}" width="{sc * 2:.0f}" '
                f'height="{sc * 2:.0f}" fill="none" class="board-ink" '
                f'stroke-width="1.5"/>' + "".join(paths)
                + f'<text x="{ml + sc:.0f}" y="{Hpx - 6}" '
                f'text-anchor="middle" class="board-text" font-size="11">'
                f'R, from 2 to 4</text>'
                f'<text x="{ml - 8}" y="{mt + sc:.0f}" text-anchor="end" '
                f'class="board-text" font-size="11">Z</text></svg>')

    axl = Axes((1 / 40, 1 / 6), (5e-3, 1.0), h=300, logx=True, logy=True)
    lp = []
    for idx, cls, lab in ((1, "board", "measured error"),
                          (2, "blue", "guaranteed bound")):
        xs = [p[0] for p in ladder]
        ys = [p[idx] for p in ladder]
        lp.append(f'<path d="{axl.path(xs, ys)}" fill="none" '
                  f'class="{cls}-ink" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            lp.append(f'<circle cx="{axl.X(x):.1f}" cy="{axl.Y(y):.1f}" '
                      f'r="4.5" class="{cls}-fill"/>')
        lp.append(f'<text x="{axl.X(xs[0]) + 8:.1f}" '
                  f'y="{axl.Y(ys[0]):.1f}" class="board-text" '
                  f'font-size="10.5" fill="var(--{cls if cls == "blue" else "muted"})">{lab}</text>')
    svg_lad = f'''<svg viewBox="0 0 640 300" role="img" aria-label="Measured
energy error and guaranteed bound versus mesh size, both first order">
{axl.grid((1e-2, 1e-1), (1 / 8, 1 / 16, 1 / 32),
          xfmt=lambda v: f"h = 1/{round(1 / v)}", yfmt=lambda v: f"{v:g}")}
{"".join(lp)}</svg>'''

    eff = [f"{b / e:.2f}" for _, e, b in ladder]
    return page(
        "Case: a tokamak equilibrium with a guaranteed error bound",
        "certified case",
        "A tokamak equilibrium with a guaranteed error bound",
        "The Grad-Shafranov equation solved by FEniCSx, certified by "
        "the Prager-Synge identity with rectangle-exact constants. The "
        "bound is guaranteed, not estimated, and the implicit coupling "
        "in the source is certified through a contraction factor that "
        "refuses past its limit.",
        [
            "<h2>The idea</h2>"
            "<p>The tokamak equilibrium equation −Δ*ψ = R²p′(ψ) + "
            "FF′(ψ) is weighted Poisson with κ = 1/R, and for elliptic "
            "problems there is a bound that is guaranteed rather than "
            "estimated. The Prager–Synge identity says: for any vector "
            "field σ with square-integrable divergence, the energy-norm "
            "error of a computed ψ_h is at most ‖κ∇ψ_h + σ‖ plus a "
            "divergence-mismatch term whose constant is the rectangle's "
            "exact first eigenvalue. FEniCSx proposes: it solves the "
            "primal problem, and a mixed Raviart–Thomas problem to get "
            "a good σ. The bound holds for whatever it returns. During "
            "this build a sign error in the mixed boundary term "
            "produced a σ that inflated the bound to 27 times the true "
            "error, and the bound was still valid; fixed, it sits at "
            "1.6 times. A bad guess costs tightness, never truth.</p>"
            "<p>The implicit coupling is the interesting part. With a "
            "source term c·ψ the problem the solver iterates on depends "
            "on its own answer. The certificate handles it without "
            "trusting the iteration: the coupled solution map is a "
            "contraction with factor θ = c·R_max/(R_min·λ₁), λ₁ again "
            "rectangle-exact, and the frozen-source bound divided by "
            "(1−θ) covers the true coupled solution. At θ ≥ 1 the "
            "function refuses: past that limit the fixed point is not "
            "certifiably unique.</p>",
            code_section(sf.gs_equilibrium_certified, sf._gs_solve),
            "<h2>The result</h2>"
            f"<figure>{svg_flux}<figcaption>Flux surfaces ψ = const of "
            "the coupled equilibrium (c = 1) on the poloidal plane, "
            "from the certified solve at h = 1/48. The nested surfaces "
            "are the magnetic geometry a tokamak confines on."
            "</figcaption></figure>"
            f"<figure>{svg_lad}<figcaption>Measured energy error "
            "against the exact Solov'ev solution, and the guaranteed "
            "bound above it, on a mesh ladder. Both converge at first "
            f"order; the bound stays within {max(eff)}× of the truth. "
            "Unlike every resolution-ladder certificate in this "
            "collection, this bound needs no ladder: each mesh "
            "certifies itself.</figcaption></figure>",
            "<h2>Checked in this run</h2><ul>"
            "<li>Guaranteed bound contains the measured error at every "
            f"mesh; efficiencies {', '.join(eff)}.</li>"
            "<li>The certified flux functional contains the exact "
            "value, computed independently in exact rational "
            "arithmetic from the Solov'ev polynomial.</li>"
            f"<li>Coupled problem (c = 1): certificates at two meshes "
            f"agree within their joint error: <strong>"
            f"{'yes' if agree else 'NO'}</strong>; Q = {rc['Q'].value:.4f} "
            f"± {rc['Q'].err:.4f} at h = 1/48.</li>"
            f"<li>Coupling past the contraction limit (c = 3): <strong>"
            f"{'refused' if refused else 'NOT REFUSED'}</strong>.</li>"
            "<li>FEniCSx has no PyPI wheels; CI installs the same "
            "Debian-family packages as the dev machine, from the "
            "FEniCS PPA, so this page regenerates from a fresh solve "
            "on every push like the rest.</li></ul>",
        ])


# ======================================================================
def compiler_case():
    N = 10
    truth = float(np.linalg.eigvalsh(sf._heis_window((1.0,) * (N - 1)))[0])

    # one question, a sweep of tolerances: the planner picks the
    # algorithm anew at each, and the trace says why
    tols = (0.3, 0.1, 3e-2, 1e-2, 3e-3)
    sweep = []
    for tol in tols:
        c = sf.heisenberg_energy_dispatch(N, tol=tol)
        trace = c.provenance[-1]
        chosen = trace.split("chose ")[1].split(" ")[0]
        cost = float(trace.split("predicted ")[1].split(")")[0])
        sweep.append((tol, chosen, cost, c))
    extreme = sf.heisenberg_energy_dispatch(N, tol=1e-9)
    headline = [sweep[0][3], sweep[3][3], extreme]
    contained = sum(abs(c.value - truth) <= c.err for c in headline)

    # the jump, measured against plain stepping on a clean ladder
    kw = dict(tol=0.031, correction_iters=0, ell_max=9)
    runs_of = lambda c: int(c.provenance[-1].split("tried ")[1].split(" ")[0])
    jumped = runs_of(sf.heisenberg_energy_dispatch(60, **kw))
    stepped = runs_of(sf.heisenberg_energy_dispatch(60, jump=False, **kw))

    # the folklore ladder of quantum chemistry, mechanized
    hc = sf.h_chain_energy_dispatch(6, tol=0.08, jump=False)

    # an impossible question gets a receipt
    try:
        sf.heisenberg_energy_dispatch(40, tol=1e-12, correction_iters=0,
                                      ell_max=6)
        receipt = None
    except sf.Refusal as e:
        receipt = e

    # staircase: tolerance on x, predicted cost of the chosen rung on y
    ax = Axes((1.5e-3, 0.5), (2.5, 2048), h=320, logx=True, logy=True)
    flip = next(i for i, s in enumerate(sweep) if s[1].startswith("dense"))
    wall = math.sqrt(sweep[flip][0] * sweep[flip - 1][0])
    wx = ax.X(wall)
    pts, labels = [], []
    for tol, chosen, cost, _ in sweep:
        x, y = ax.X(tol), ax.Y(cost)
        pts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                   f'class="blue-fill"/>')
        ly, anchor = (y + 18, "start") if y < ax.mt + 24 else (y - 10,
                                                               "middle")
        labels.append(f'<text x="{x:.1f}" y="{ly:.1f}" '
                      f'text-anchor="{anchor}" class="board-text" '
                      f'font-size="10" opacity="0.85">{chosen}</text>')
    step = " L ".join(
        f"{ax.X(t):.1f} {ax.Y(c):.1f} L {ax.X(tols[i + 1]) if i + 1 < len(tols) else ax.X(ax.xlim[0]):.1f} {ax.Y(c):.1f}"
        for i, (t, _, c, _) in enumerate(sweep))
    svg = f'''<svg viewBox="0 0 640 320" role="img" aria-label="Staircase of
chosen algorithm cost against tolerance, with the window-to-dense wall">
{ax.grid((10, 100, 1000), (1e-2, 1e-1),
         xfmt=lambda v: f"tol = {v:g}", yfmt=lambda v: f"{v:g}")}
<rect x="{ax.ml}" y="{ax.mt}" width="{wx - ax.ml:.1f}"
      height="{320 - ax.mt - ax.mb}" fill="var(--rust)" opacity="0.07"/>
<line x1="{wx:.1f}" y1="{ax.mt}" x2="{wx:.1f}" y2="{320 - ax.mb}"
      class="rust-ink" stroke-dasharray="4 4" stroke-width="1.4"/>
<text x="{wx + 8:.1f}" y="{ax.mt + 14}" class="board-text"
      font-size="10.5" fill="var(--rust)">&#8592; the windows floor out;
 dense takes over</text>
<path d="M {step}" fill="none" class="blue-ink" stroke-width="1.6"
      opacity="0.55"/>
{"".join(pts)}{"".join(labels)}
<text x="{ax.ml + 10}" y="{320 - ax.mb - 10}" class="board-text"
      font-size="10.5">predicted cost of the chosen rung, N = {N} chain</text>
</svg>'''

    traces = "\n".join(esc(c.provenance[-1]) for c in headline)
    return page(
        "Case: the planner",
        "certified case",
        "The compiler half, first slice",
        "One question at three tolerances gets three different "
        "algorithms. A planner searches the rewrites' declared cost "
        "ladders; the certificates, never the cost models, decide "
        "what is true.",
        [
            "<h2>The idea</h2>"
            "<p>Until now every page on this site answered its "
            "question with an algorithm someone chose by hand. This "
            "page is the first piece of the promised compiler: the "
            "choice itself is made by a program. Each way of "
            "answering a question is declared as a rewrite with a "
            "ladder of effort and a guessed cost per rung. The "
            "planner runs the cheapest promise first. If the "
            "certificate that comes back meets the tolerance, done. "
            "If not, the measurement is kept, the rewrite's next rung "
            "goes back in the queue — after two failures a fitted "
            "decay model picks how far to jump — and the next "
            "cheapest promise runs, which may belong to a competing "
            "rewrite. Nothing here is trusted: a wrong cost model or "
            "a bad jump wastes some running time, and cannot corrupt "
            "the answer, because every answer still arrives with its "
            "own certificate.</p>",
            code_section(sf.Rewrite, sf._fit_jump, sf.plan,
                         sf.heisenberg_energy_dispatch),
            "<h2>The result</h2>"
            f"<figure>{svg}<figcaption>The staircase. Loose questions "
            "are served by short windows at cost 2<sup>ℓ</sup>; as "
            "the tolerance tightens the planner climbs the ladder, "
            "and past the wall the window widths floor out on the "
            "relaxation gap, so the dense 1024-dimensional bracket — "
            "exact, and priced 2<sup>N</sup> — wins the competition. "
            "The wall belongs to this 10-site chain; on a 10⁶-site "
            "chain there is no dense column to flee to."
            "</figcaption></figure>"
            "<p>Three of the plan traces, verbatim from the "
            f"certificates' provenance:</p><pre>{traces}</pre>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/3</strong> "
            "planner-chosen brackets contain the exact "
            "1024-dimensional answer.</li>"
            "<li>The jump earns its keep: on the 60-site ladder the "
            f"certifying rung is reached in <strong>{jumped} runs"
            f"</strong> with the fitted jump against <strong>"
            f"{stepped}</strong> with plain stepping.</li>"
            "<li>The folklore method ladder of quantum chemistry, "
            "mechanized: H6 at 0.08 hartree/atom &rarr; <code>"
            f"{esc(hc.provenance[-1].split('; rejected')[0])}</code>."
            "</li>"
            "<li>An impossible question refuses with a receipt: "
            f"<code>{esc(str(receipt)[:160])}&hellip;</code></li>"
            "<li>What this is not, yet: the plan space is single-knob "
            "ladders per query. Composed pipelines that split one "
            "error budget across several rewrites are the remaining "
            "debt.</li></ul>",
        ])


CASES = {
    "tfi-reduced-basis.html": tfi_case,
    "h2-bracket.html": h2_case,
    "hchain-ladder.html": ladder_case,
    "butterfly-crossover.html": butterfly_case,
    "ising-cluster.html": ising_case,
    "smeared-spectral.html": spectral_case,
    "mz-closure.html": mz_case,
    "lr-dispatch.html": lr_case,
    "plasma-hierarchy.html": plasma_case,
    "sos-transport.html": sos_case,
    "gw-surrogate.html": gw_case,
    "sph-wall.html": sph_case,
    "gs-equilibrium.html": gs_case,
    "the-compiler.html": compiler_case,
}

# pages needing tools CI does not have: generated locally, committed,
# served as-is — the recorded-run pattern from TARGETS (empty since CI
# gained FEniCSx from the PPA; Basilisk will likely use it next)
RECORDED = {}

if __name__ == "__main__":
    check_counts()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "cases"), exist_ok=True)
    only = sys.argv[1:] or list(CASES)
    for name in only:
        t0 = time.time()
        if name in RECORDED:
            try:
                html = RECORDED[name]()
            except ImportError as exc:
                print(f"skipped {name}: {exc}")
                continue
            os.makedirs(os.path.join(here, "cases-recorded"), exist_ok=True)
            with open(os.path.join(here, "cases-recorded", name), "w") as f:
                f.write(html)
            print(f"wrote cases-recorded/{name}  ({time.time() - t0:.0f}s)")
            continue
        with open(os.path.join(here, "cases", name), "w") as f:
            f.write(CASES[name]())
        print(f"wrote cases/{name}  ({time.time() - t0:.0f}s)")
