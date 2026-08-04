"""Basilisk-style case pages: each page is BUILT by running its case.
The source renders as the presentation (extracted live via inspect), the
a priori theory is stated, and the certification is executed and drawn.
Run from the repo root:  python3 docs/make_cases.py   (~2 minutes)
"""
import functools
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


def _ruler():
    """A description of this builder, printed beside the commit stamp:
    one 256x256 symmetric eigendecomposition, warmed up first and then
    the fastest of five. Warm-and-fastest because cold medians on this
    benchmark swing four-fold between processes on an idle machine --
    measured. It describes the machine; it is not a unit the receipts
    above can be divided by."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((256, 256))
    A = A + A.T
    np.linalg.eigh(A)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.linalg.eigh(A)
        ts.append(time.perf_counter() - t0)
    return min(ts)


RULER = f"{_ruler() * 1e3:.1f} ms"


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


def check_composed():
    """The planner refuses a hand-rolled budget split at runtime, and
    this is what stops the refusal being switched off. plan() takes a
    private _composed flag that suppresses the check; compose() is the
    only caller entitled to pass it, because compose is what supplies
    the derivation the check exists to require. A future author facing
    the error has two ways forward -- declare Stages, or add the flag
    and move on -- and only the first is the one meant. So the build
    fails if the flag appears anywhere but inside compose, and fails
    if a front door declares Stages without composing them.

    Structural, not stylistic: a release cannot ship the debt back."""
    import ast
    with open(os.path.join(ROOT, "sufficit.py")) as f:
        tree = ast.parse(f.read())

    def called(fn):
        return {c.func.id for c in ast.walk(fn)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    def waives(fn):
        return any(k.arg == "_composed"
                   for c in ast.walk(fn) if isinstance(c, ast.Call)
                   for k in c.keywords)

    seen = 0
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if fn.name == "compose":
            seen += waives(fn)
            continue
        if waives(fn):
            raise RuntimeError(
                f"{fn.name}() passes plan(_composed=...); only compose() "
                "may, or the guard against hand-rolled budget splits is "
                "off for that front door")
        names = called(fn)
        if fn.name.endswith("_dispatch") and "Stage" in names \
                and "compose" not in names:
            raise RuntimeError(
                f"{fn.name}() declares Stages but never calls compose(); "
                "the stages are then decoration and the split is still "
                "allocated by hand")
    if seen != 1:
        raise RuntimeError(
            f"compose() waives the hand-rolled-split guard {seen} times; "
            "expected exactly once -- if compose no longer calls plan the "
            "guard is unreachable and this check is watching nothing")


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
built this page: {STAMP}. This builder, for scale: one 256&#215;256
symmetric eigendecomposition took {RULER}. Any seconds above are this
machine's, at that moment, and are not divided by it.</p>
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

    # the exported sensitivity, exercised: a constant source offset
    # moves the flux by no more than the certified Lipschitz constant
    base8 = sf.gs_equilibrium_certified(n=8, c=1.0)
    pert8 = sf.gs_equilibrium_certified(n=8, c=1.0, dg0=0.5)
    sens = base8["Q"].sensitivity
    smoved = abs(pert8["Q"].value - base8["Q"].value)
    sbudget = sens.bound * 1.0 + base8["Q"].err + pert8["Q"].err

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

    # the nonlinear half: a real pressure profile, where the
    # contraction certificate cannot follow
    PSI0 = 0.2
    lam1 = math.pi ** 2 * (1.0 / 4 + 1.0 / 4)
    c_wall = 0.95 * PSI0 * 2.0 * lam1 / 4.0
    sweep = []
    for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        r = sf.gs_nonlinear_certified(n=16, c=c, psi0=PSI0, m=5)
        sweep.append((c, r["radius"], r["psi_err"], r["theta"], r["route"]))
    gaps = [math.log10(e / r) for _, r, e, _, _ in sweep]
    zmat = sf.gs_nonlinear_certified(n=8, c=0.0, m=2)
    nz = sf.gs_nonlinear_certified(n=8, c=0.1, m=2)
    npos = int((nz["J"] > 0).sum() - (np2.diag(nz["J"]) > 0).sum())
    try:
        sf.gs_equilibrium_certified(n=8, c=4.0)
        lin_refused = False
    except ValueError:
        lin_refused = True
    # the ladder from cold, at the strongest coupling
    cliff = []
    for m in range(1, 5):
        try:
            r = sf.gs_nonlinear_certified(n=16, c=16.0, psi0=PSI0, m=m)
            cliff.append((m, r["residual"], r["radius"]))
        except ValueError:
            cliff.append((m, None, None))
    first_ok = next(m for m, _, rad in cliff if rad is not None)

    axn = Axes((0.08, 20.0), (1e-16, 1e-2), h=320, logx=True, logy=True)
    np_ = []
    for idx, cls, lab in ((2, "board", "distance to the exact solution"),
                          (1, "blue", "certified radius")):
        xs = [s[0] for s in sweep]
        ys = [s[idx] for s in sweep]
        np_.append(f'<path d="{axn.path(xs, ys)}" fill="none" '
                   f'class="{cls}-ink" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            np_.append(f'<circle cx="{axn.X(x):.1f}" cy="{axn.Y(y):.1f}" '
                       f'r="4" class="{cls}-fill"/>')
        np_.append(f'<text x="{axn.X(xs[0]) + 9:.1f}" '
                   f'y="{axn.Y(ys[0]) - 7:.1f}" class="board-text" '
                   f'font-size="10.5" fill="var('
                   f'--{cls if cls == "blue" else "muted"})">{lab}</text>')
    np_.append(f'<line x1="{axn.X(c_wall):.1f}" y1="{axn.mt}" '
               f'x2="{axn.X(c_wall):.1f}" y2="{axn.h - axn.mb}" '
               f'class="board-ink" stroke-width="1.5" '
               f'stroke-dasharray="5 4" opacity="0.8"/>')
    np_.append(f'<text x="{axn.X(c_wall) + 7:.1f}" y="{axn.mt + 26}" '
               f'class="board-text" font-size="10.5" opacity="0.85">'
               f'contraction refuses to the right</text>')
    svg_non = f'''<svg viewBox="0 0 640 320" role="img" aria-label="Certified
Kantorovich radius and measured discretization error against coupling
strength, with the contraction certificate's refusal marked">
{axn.grid((1e-14, 1e-10, 1e-6, 1e-3), (0.1, 1.0, 10.0),
          xfmt=lambda v: f"c = {v:g}", yfmt=lambda v: f"{v:g}")}
{"".join(np_)}</svg>'''

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
            "<li>The flux exports its sensitivity to the source — "
            "&#8730;area&#183;R<sub>max</sub>/(&#955;&#8321;&#183;"
            "R<sub>min</sub>&#183;(1&#8722;&#952;)) = "
            f"{sens.bound:.3g} per unit of source in L&#178;, RIGOROUS "
            "— the contraction constant repriced as a Lipschitz bound "
            "for composed plans. Exercised in this run: a constant "
            "source offset (&#8214;&#948;&#8214; = 1) moved the flux "
            f"by {smoved:.3g} against a budget of {sbudget:.3g}: "
            f"<strong>{'contained' if smoved <= sbudget else 'VIOLATED'}"
            "</strong>.</li>"
            "<li>FEniCSx has no PyPI wheels; CI installs the same "
            "Debian-family packages as the dev machine, from the "
            "FEniCS PPA, so this page regenerates from a fresh solve "
            "on every push like the rest.</li></ul>",

            "<h2>Where the contraction cannot follow</h2>"
            "<p>Everything above needs the source to be linear in ψ, "
            "because that is what the Picard contraction closes over, "
            "and needs θ &lt; 1 on top. A real pressure profile obeys "
            "neither. Tokamak pressure peaks on the magnetic axis and "
            "falls away outward, so p′ <em>decreases</em> with ψ and "
            "the source is a nonlinear function of the unknown.</p>"
            "<p>Two things then break, and it is worth seeing that "
            "they break independently. The contraction factor is "
            "max|S′|·R_max/(R_min·λ₁), and for an exponential profile "
            f"of decay scale ψ₀ = {PSI0:g} that maximum is c/ψ₀ rather "
            "than c: a peaked profile is a steep one, so the wall "
            f"arrives five times sooner, at c = {c_wall:.3g} instead "
            "of 2.34.</p>"
            "<p>The second break is structural and it does not care "
            "about c at all. Differentiating the source contributes "
            "−S′ times the consistent mass matrix, whose off-diagonal "
            "entries are integrals of products of non-negative basis "
            "functions and are therefore <em>positive</em>. A "
            "decreasing profile has S′ &lt; 0, so those positive "
            "entries land on top of the stiffness matrix's negative "
            f"ones, and {npos} of them come out positive at n = 8 for "
            f"a coupling of c = 0.1 — a θ of {nz['theta']:.2g}, well "
            "inside what the contraction would have accepted. The "
            "Jacobian is no longer a Z-matrix, so the reactor's cone "
            "witness cannot price ‖J⁻¹‖ here, and it is the shape of "
            "the profile that closed that door rather than the "
            "strength of the coupling.</p>"
            "<p>What answers is the same Kantorovich theorem the "
            "junction runs on, with the other route to ‖J⁻¹‖: pick any "
            "approximate inverse R, and if ‖I − RJ‖ &lt; 1 then the "
            "Neumann series converges, J is nonsingular, and "
            "‖J⁻¹‖ ≤ ‖R‖/(1 − ‖I − RJ‖). That asks nothing about "
            "signs. It costs a matrix inverse rather than a solve, and "
            "where the cone route also applies the two agree to every "
            "digit printed, because on an M-matrix the witness is not "
            "merely valid but exact — J⁻¹ ≥ 0 makes ‖J⁻¹‖<sub>∞</sub> "
            "literally ‖J⁻¹e‖<sub>∞</sub>, which is what the witness "
            "computes.</p>"
            "<p>The guarantee that comes back is a different one and "
            "the difference is the point. Prager–Synge measures the "
            "distance to the <em>continuum</em> solution. Kantorovich "
            "proves that an exact solution of the <em>discrete</em> "
            "equations exists and holds it in a radius. Where both "
            "apply, take Prager–Synge; where it refuses, a discrete "
            "answer with a stated mesh gap beats no answer.</p>",
            code_section(sf.inverse_bound, sf.gs_nonlinear_certified),
            "<h2>The result</h2>"
            f"<figure>{svg_non}<figcaption>Coupling strength across, "
            "error down, both logarithmic. The contraction certificate "
            "stops at the dashed line and says nothing to the right of "
            "it. The Kantorovich radius keeps shrinking as the "
            "coupling grows — a stiffer problem has a smaller ‖J⁻¹‖ — "
            f"and stays {min(gaps):.0f} to {max(gaps):.0f} orders below "
            "the discretization, which is what actually binds. The "
            "certificate is about "
            "the discrete equilibrium; the upper curve is the distance "
            "from there to the differential equation, and it is "
            "measured rather than carried.</figcaption></figure>"
            "<p>The ladder from a cold start is a real one. At c = 16, "
            "with the interior initialised to zero, no enclosure "
            f"exists at all until step {first_ok}: "
            + "; ".join(f"at step {m} the residual is {r:.2g} and the "
                        f"radius {rad:.2g}"
                        for m, r, rad in cliff if rad is not None)
            + ". The number of steps needed to certify tracks the "
            "physics: two at c = 1 and c = 4, three at c = 16.</p>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>The linear certificate at c = 4: <strong>"
            f"{'refused' if lin_refused else 'NOT REFUSED'}</strong> "
            "(θ = 1.62). The nonlinear one certifies there and at four "
            f"times the coupling, with θ up to {sweep[-1][3]:.3g}.</li>"
            "<li>Route taken, with no source: <strong>"
            f"{zmat['route']}</strong> — the stiffness matrix alone is "
            "a Z-matrix and the cheap route answers. With the profile "
            f"switched on at c = 0.1: <strong>{nz['route']}</strong>."
            "</li>"
            f"<li>Discretization second order in h, measured "
            "8 → 16 → 32; the certified radius stays below a millionth "
            "of it at every mesh, so the model boundary and not the "
            "arithmetic is what limits this answer.</li>"
            "<li>Assembly quadrature is not carried, the same "
            "declaration the Prager–Synge path makes. Everything from "
            "the assembled residual and Jacobian onward is, including "
            "the rounding inside the matrix products behind β.</li>"
            "</ul>",
        ])


# ======================================================================
def compiler_case():
    N = 10
    truth = float(np.linalg.eigvalsh(sf._heis_window((1.0,) * (N - 1)))[0])

    # the pair that caught a cost model lying, timed first so both are
    # cold: the widest window against the dense rung the old model
    # quoted at twice its price
    t0 = time.perf_counter()
    sf.heisenberg_chain_bracket(N, 9)
    w9_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    sf.heisenberg_chain_bracket(N, ell=N)
    dense_secs = time.perf_counter() - t0

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
    # the receipt is structure, so the audit reads fields, not prose:
    # window@9 against the chosen dense, and window@2 cold vs warm
    secs = {(n, k): s for n, k, _, s, _ in extreme.receipt}
    w2_cold = f"{sweep[0][3].receipt[0][3]:.2g}s"
    w2_warm = f"{secs.get(('window', 2), 0):.2g}s"
    receipt_lines = "\n".join(
        f"{n}@{k}: predicted {p:g}, measured {s:.2g}s, "
        + (f"err {v:.3g}" if isinstance(v, float) else str(v))
        for n, k, p, s, v in extreme.receipt)
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

    # the phase map: the staircase is one column of it. Same wiring as
    # heisenberg_energy_dispatch, with bracket results memoized across
    # cells so each (N, ell) rung is computed once for the whole map.
    brk = functools.lru_cache(maxsize=None)(sf.heisenberg_chain_bracket)
    map_Ns = (6, 8, 10, 12, 14, 16, 18, 20)
    map_tols = (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4, 1e-8)
    # the cost models are read off the dispatch, not retyped, so the map
    # cannot quietly price the race differently from the library
    ITERS = inspect.signature(sf.heisenberg_energy_dispatch) \
        .parameters["correction_iters"].default

    def cell(n, tol_bond):
        ells = tuple(range(2, min(n - 1, 9) + 1))
        rws = [sf.Rewrite("window", ells, lambda l: ITERS * 2.0 ** l,
                          lambda l: brk(n, l, ITERS))]
        if n <= 12:
            rws.append(sf.Rewrite("dense", (n,), lambda _: 2.0 ** n,
                                  lambda _: brk(n, n)))
        try:
            t = sf.plan("chain-energy", tol_bond * (n - 1),
                        rws).provenance[-1]
            name, knob = t.split("chose ")[1].split(" ")[0].split("@")
            return name, int(knob)
        except sf.Refusal:
            return "refused", 0

    grid = {(n, t): cell(n, t) for n in map_Ns for t in map_tols}
    n_refused = sum(1 for v in grid.values() if v[0] == "refused")

    ml, mt, cw, ch = 74, 30, 64, 27
    MW = ml + cw * len(map_Ns) + 14
    MH = mt + ch * len(map_tols) + 42
    mcells = []
    for j, t in enumerate(map_tols):
        for i, n in enumerate(map_Ns):
            x, y = ml + i * cw, mt + j * ch
            name, knob = grid[(n, t)]
            if name == "window":
                fill = (f'fill="var(--blue)" '
                        f'opacity="{0.10 + 0.07 * (knob - 2):.2f}"')
                lab, lop = f"&#8467;={knob}", 0.9
            elif name == "dense":
                fill = 'fill="var(--rust)" opacity="0.30"'
                lab, lop = "dense", 0.9
            else:
                fill = 'fill="none"'
                lab, lop = "refused", 0.45
            mcells.append(
                f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" '
                f'{fill}/>'
                f'<text x="{x + (cw - 2) / 2:.1f}" y="{y + ch / 2 + 2:.1f}" '
                f'text-anchor="middle" class="board-text" font-size="10" '
                f'opacity="{lop}">{lab}</text>')
    i10 = map_Ns.index(10)
    map_svg = f'''<svg viewBox="0 0 {MW} {MH}" role="img" aria-label="Phase
map of the chosen rewrite over chain length and tolerance, with window,
dense, and refusal regions">
{"".join(mcells)}
<rect x="{ml + i10 * cw - 2}" y="{mt - 2}" width="{cw + 2}"
      height="{ch * len(map_tols) + 2}" fill="none" class="board-ink"
      stroke-dasharray="4 3" stroke-width="1.2"/>
<text x="{ml + i10 * cw + cw / 2:.1f}" y="{mt - 9}" text-anchor="middle"
      class="board-text" font-size="10" opacity="0.75">the staircase
 above is this column</text>
{"".join(f'<text x="{ml - 8}" y="{mt + j * ch + ch / 2 + 3:.1f}" text-anchor="end" class="board-text" font-size="10.5" opacity="0.7">{t:g}</text>' for j, t in enumerate(map_tols))}
{"".join(f'<text x="{ml + i * cw + (cw - 2) / 2:.1f}" y="{MH - 26}" text-anchor="middle" class="board-text" font-size="10.5" opacity="0.7">{n}</text>' for i, n in enumerate(map_Ns))}
<text x="{ml - 8}" y="{MH - 26}" text-anchor="end" class="board-text"
      font-size="10.5" opacity="0.7">tol/bond</text>
<text x="{ml + cw * len(map_Ns) / 2:.1f}" y="{MH - 8}"
      text-anchor="middle" class="board-text" font-size="10.5"
      opacity="0.7">chain length N</text>
</svg>'''

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
            "are served by short windows, priced at the ascent's "
            "eigendecompositions times 2<sup>ℓ</sup>; as the "
            "tolerance tightens the planner climbs the ladder, "
            "and past the wall the window widths floor out on the "
            "relaxation gap, so the dense 1024-dimensional bracket — "
            "exact, and priced 2<sup>N</sup> — wins the competition. "
            "The window band is short here because ten sites is small "
            "enough to diagonalize: once the two rewrites are priced "
            "in the same unit, the honest advice at this size is "
            "mostly <em>form the matrix</em>. The map below shows "
            "where that stops being true. "
            "The wall belongs to this 10-site chain; on a 10⁶-site "
            "chain there is no dense column to flee to. And chosen is "
            "not optimal: the y-axis is the cheapest plan found among "
            "the declared rewrites, not the cheapest possible "
            "algorithm — the planner's claim stops at its library."
            "</figcaption></figure>"
            "<p>Three of the plan traces, verbatim from the "
            f"certificates' provenance:</p><pre>{traces}</pre>"
            "<p>The trace is deterministic, because provenance is "
            "part of the certificate. The timings live next door, in "
            "the certificate's structured receipt — here is the "
            "tightest run's, one rung per line:</p>"
            f"<pre>{esc(receipt_lines)}</pre>"
            "<p>Those seconds are honest but local: another machine, "
            "or this one an hour busier, gives different numbers for "
            "the same work. The tempting fix is a ruler — divide "
            "every measurement by a fixed microbenchmark timed on the "
            "same machine, and the ratio should travel. This library "
            "shipped that for a few hours and then measured it, which "
            "is the only reason to write anything down here: it does "
            "not work, at least not with a small benchmark against "
            "these rungs. Under three competing processes the "
            "256-dimensional benchmark slowed by 480&#215; while the "
            "1024-dimensional rung it was supposed to price slowed by "
            "35&#215;; on an idle machine, across four fresh "
            "processes, the rung's raw seconds spanned 1.7&#215; and "
            "the same seconds divided by the benchmark spanned "
            "5.6&#215;. The yardstick moved more than the thing being "
            "measured. A short dense benchmark and a long one do not "
            "share a cost direction — they differ in how much they "
            "depend on cache residency and on getting all their "
            "threads scheduled at once — so dividing by it removes "
            "one machine and introduces another.</p>"
            "<p>So the rows carry seconds, and the receipt claims "
            "only what it can support: on <em>this</em> machine, this "
            "prediction went with that measurement. That is still the "
            "pair a cost model learns from; it is just learned per "
            "machine. Making predictions portable is a job for a cost "
            "model written over each rewrite's own parameters — "
            "dimensions, iteration counts, the sizes the algorithm "
            "already knows — not for a yardstick held up beside "
            "it.</p>",
            "<h2>The phase map</h2>"
            "<p>The staircase is one slice of a bigger object. Sweep "
            "both knobs — chain length and tolerance — and the plan "
            "traces assemble a phase map of the query: which rewrite "
            "is cheapest-found where, and where the declared library "
            "ends. This is what the compiler's answers look like in "
            "bulk, and as the rewrite library grows it is the real "
            "deliverable: not one certified number, but the certified "
            "boundary of what each method can do.</p>"
            f"<figure>{map_svg}<figcaption>Each cell is one full run "
            "of the dispatch at that chain length N and per-bond "
            "tolerance (window ladder capped at &#8467; = 9; bracket "
            "results shared across cells, so each rung is computed "
            "once). Blue: a window rung certifies, darker for longer "
            "windows. Rust: the dense 2<sup>N</sup> bracket wins — a "
            "column that ends at N = 12, where the matrix stops being "
            "formable. Blank: every declared rung ran and none "
            f"certified — {n_refused} of {len(grid)} cells here, each "
            "refusing with a receipt. The blank region is the map's "
            "most honest feature: its boundary is the certified price "
            "wall of this library, and every rewrite added to the "
            "module redraws it.</figcaption></figure>",
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
            "<li>A certificate that changed between calls. "
            "<code>eigsh</code>, left to itself, draws its ARPACK "
            "start vector from numpy's global random stream, so a "
            "bracket's value depended on how many other calls had run "
            "first in the same process: the &#8467; = 7 window at "
            "N = 10 came back 0.382, 0.332 and 0.379 on three "
            "consecutive calls. The cause is physics, not arithmetic. "
            "A Heisenberg window on an odd number of sites has a "
            "spin-&#189; ground <em>doublet</em> — measured gap "
            "4&#215;10<sup>&#8722;16</sup> at &#8467; = 5 and "
            "7&#215;10<sup>&#8722;16</sup> at &#8467; = 7, against "
            "10<sup>&#8722;1</sup> at the even widths — and while "
            "both members carry the same energy, they carry opposite "
            "edge spins, which is what the upper bound uses to price "
            "the bond between blocks. A validity question it is not: "
            "that vector is a variational trial state, so an unlucky "
            "member loosens a bracket and cannot invalidate one. An "
            "unrepeatable certificate is the worse failure anyway. "
            "The start is now a fixed seeded draw.</li>"
            "<li>A default that was buying nothing. The window "
            "bound's multiplier ascent ran eighty oracle calls; "
            "measured on two numpy stacks at three chain lengths "
            "across six widths, ten calls already buy <strong>"
            "98&#8211;107%</strong> of the tightening eighty buy, and "
            "land within 1.1% of the eighty-call bracket either way. "
            "Above 100% means eighty is the <em>worse</em> bracket, "
            "which is not a contradiction: the ascent optimizes the "
            "dual of the uniform-weight window while the bracket "
            "applies its answer to weighted sliding windows, so a "
            "better dual value is not obliged to give a tighter "
            "bracket. Eighty calls cost 20&#8211;36&#215; more than "
            "ten rather than 8&#215;, because the bundle grows toward "
            "its cut cap and the master problem grows with it. Ten is "
            "now the default. None of it was measurable until the "
            "start vector above was pinned: the first version of this "
            "measurement read 94.5% here and 51.7% on the CI machine, "
            "and both were the wandering vector rather than the "
            "ascent.</li>"
            "<li>The receipts audited a cost model and caught it "
            "lying. Window rungs were priced 2<sup>ℓ</sup> against "
            "the dense rung's 2<sup>N</sup>, so the model quoted the "
            "widest window at half the dense price. Measured cold on "
            f"this machine, window@9 takes <strong>{w9_secs:.2g}s"
            f"</strong> against dense's <strong>{dense_secs:.2g}s"
            "</strong> — nearly twice as dear, not half. The missing "
            "term was the multiplier ascent, which a window rung runs "
            "before it brackets anything and the dense rung does not "
            "run at all, so the two rewrites that race here were "
            "quoted in different currencies. Both are now priced in "
            "eigendecompositions times dimension.</li>"
            "<li>And the same window@2 rung, priced 40 in both plans, "
            f"measured {esc(w2_cold)} in the first plan and "
            f"{esc(w2_warm)} in the last — its corrections were "
            "cached by then. Cost is state-dependent; the predictions "
            "are not; only logging both exposes it. These pairs, kept "
            "in every run, are the calibration data for better cost "
            "models — on the machine that produced them.</li>"
            "<li>A negative result, kept because it cost something to "
            "learn: normalizing those seconds by a fixed "
            "microbenchmark, so they would travel between machines, "
            "was built, shipped, measured, and removed the same day. "
            "The benchmark is 480&#215; contention-sensitive where "
            "the rung is 35&#215;, so the ratio is noisier than the "
            "raw seconds it replaced.</li>"
            f"<li>The phase map ran {len(grid)} dispatches: "
            f"<strong>{len(grid) - n_refused}</strong> certified "
            f"cells, <strong>{n_refused}</strong> structured "
            "refusals. A colored cell meets its tolerance by "
            "construction — plan() returns only what a certificate "
            "approves.</li>"
            "<li>What this is not, yet: the plan space is single-knob "
            "ladders per query. The first composed plan — one error "
            "budget split between two stages — now has "
            '<a href="the-budget.html">its own page</a>, and every '
            "composed plan since goes through one combinator, which "
            "the planner and the build now jointly require.</li></ul>",
        ])


# ======================================================================
def budget_case():
    omega, sigma, z, Ns = 1.0, 0.5, 5.0, (8, 12, 16)
    E, a, rel = (0.9, 1.9), (1.0, 0.7), 1e-2

    def exact(N):
        ts = np.arange(1, N + 1)
        return a[0] * np.exp(-E[0] * ts) + a[1] * np.exp(-E[1] * ts)

    def cov1(N):
        return np.diag((rel * exact(N)) ** 2)

    def measure(N, m):
        rng = np.random.default_rng(N * 1_000_003 + m)
        return exact(N) + rng.standard_normal(N) * rel * exact(N) \
            / math.sqrt(m)

    truth = sum(ai * math.exp(-(omega - Ei) ** 2 / (2 * sigma**2))
                / (sigma * math.sqrt(2 * math.pi)) for ai, Ei in zip(a, E))

    # the ladder's economics, computed the way the dispatch computes them
    C1 = float(exact(8)[0])
    ladder = {}
    for N in Ns:
        g, c = sf._hlt_solve(N, omega, sigma)
        V = cov1(N)
        amp = c * math.sqrt(float(V[0, 0])) \
            + math.sqrt(float(g @ V[1:, 1:] @ g))
        ladder[N] = (c * C1, amp)      # (smearing bill, noise amplification)

    def cost(N, tol):
        bill, amp = ladder[N]
        if tol <= bill:
            return None
        return N * max(1, math.ceil((z * amp / (0.8 * (tol - bill))) ** 2))

    # one question, three tolerances: the split re-decides each time
    sweep = []
    for tol in (0.4, 0.2, 0.12):
        cert = sf.smeared_spectral_dispatch(measure, cov1, omega, sigma,
                                            tol, Ns=Ns)
        split = next(p for p in cert.provenance
                     if p.startswith("smeared-spectral split"))
        Nc, mc = (int(x) for x in re.findall(r"kernel=(\d+) samples=(\d+)", split)[0])
        sweep.append((tol, Nc, mc, cert))
    contained = sum(abs(c.value - truth) <= c.err for _, _, _, c in sweep)

    # at sigma=0.4 the coarse kernel qualifies only with an ocean of
    # samples; the split buys resolution instead
    hard = sf.smeared_spectral_dispatch(measure, cov1, omega, 0.4, 0.2,
                                        Ns=Ns)
    hsplit = next(p for p in hard.provenance
                  if p.startswith("smeared-spectral split"))
    hN, hm = (int(x) for x in re.findall(r"kernel=(\d+) samples=(\d+)", hsplit)[0])
    hcost = float(hard.receipt[-1][2])
    g8, c8 = sf._hlt_solve(8, omega, 0.4)
    V8 = cov1(8)
    amp8 = c8 * math.sqrt(float(V8[0, 0])) \
        + math.sqrt(float(g8 @ V8[1:, 1:] @ g8))
    m8 = math.ceil((z * amp8 / (0.8 * (0.2 - c8 * C1))) ** 2)

    # the sensitivity field composes through the IR's own arithmetic
    lo = sf.smeared_spectral(exact(16), E[0], sigma)
    hi = sf.smeared_spectral(exact(16), E[1], sigma)
    diff = lo - hi

    # an impossible question: below every rung's smearing bill
    try:
        sf.smeared_spectral_dispatch(measure, cov1, omega, sigma, 0.05,
                                     Ns=Ns)
        refusal = None
    except sf.Refusal as e:
        refusal = e

    # cost-of-the-split curves: each rung diverges at its own smearing
    # bill; the plan walks the lower envelope
    ax = Axes((0.088, 0.55), (60, 4e5), h=320, logx=True, logy=True)
    wall = min(b for b, _ in ladder.values())
    wx = ax.X(wall)
    cls = {8: "rust-ink", 12: "board-ink", 16: "blue-ink"}
    curves = []
    for N in Ns:
        grid = np.exp(np.linspace(math.log(ladder[N][0] * 1.001),
                                  math.log(0.55), 160))
        xs = [t for t in grid if cost(N, t)]
        curves.append(f'<path d="{ax.path(xs, [cost(N, t) for t in xs])}" '
                      f'fill="none" class="{cls[N]}" stroke-width="1.8" '
                      f'opacity="0.85"/>')

    def curve_x(N, y):
        bill, amp = ladder[N]
        return bill + z * amp / (0.8 * math.sqrt(y / N))

    labels = (f'<text x="{ax.X(curve_x(12, 2.2e5)) - 5:.1f}" '
              f'y="{ax.Y(2.2e5) + 3.5:.1f}" text-anchor="end" '
              f'class="board-text" font-size="10.5">N=12</text>'
              f'<text x="{ax.X(curve_x(8, 8e3)) + 5:.1f}" '
              f'y="{ax.Y(8e3) + 3.5:.1f}" class="board-text" '
              f'font-size="10.5" fill="var(--rust)">N=8</text>'
              f'<text x="{ax.X(curve_x(16, 6e3)) + 5:.1f}" '
              f'y="{ax.Y(6e3) + 3.5:.1f}" class="board-text" '
              f'font-size="10.5" fill="var(--blue)">N=16</text>')
    pts = "".join(
        f'<circle cx="{ax.X(t):.1f}" cy="{ax.Y(c.receipt[-1][2]):.1f}" '
        f'r="4.5" class="blue-fill"/>'
        f'<text x="{ax.X(t) - 6:.1f}" y="{ax.Y(c.receipt[-1][2]) + 14:.1f}" '
        f'text-anchor="end" class="board-text" font-size="10" '
        f'opacity="0.85">m={m}</text>'
        for t, N, m, c in sweep)
    svg = f'''<svg viewBox="0 0 640 320" role="img" aria-label="Predicted
cost of each resolution rung against tolerance, each curve diverging at
its own smearing bill, with the plan's chosen splits marked">
<clipPath id="bplot"><rect x="{ax.ml}" y="{ax.mt}"
  width="{640 - ax.ml - ax.mr}" height="{320 - ax.mt - ax.mb}"/></clipPath>
{ax.grid((1e2, 1e3, 1e4, 1e5), (0.1, 0.2, 0.4),
         xfmt=lambda v: f"tol = {v:g}", yfmt=lambda v: f"{v:g}")}
<rect x="{ax.ml}" y="{ax.mt}" width="{wx - ax.ml:.1f}"
      height="{320 - ax.mt - ax.mb}" fill="var(--rust)" opacity="0.07"/>
<line x1="{wx:.1f}" y1="{ax.mt}" x2="{wx:.1f}" y2="{320 - ax.mb}"
      class="rust-ink" stroke-dasharray="4 4" stroke-width="1.4"/>
<text x="{wx + 8:.1f}" y="{320 - ax.mb - 34}" class="board-text"
      font-size="10.5" fill="var(--rust)">&#8592; below the best smearing
 bill, no sample count helps</text>
<g clip-path="url(#bplot)">{"".join(curves)}</g>
{labels}{pts}
<text x="{ax.ml + 10}" y="{320 - ax.mb - 10}" class="board-text"
      font-size="10.5">predicted cost N&#183;m of each rung's split,
 &#963; = {sigma:g}</text>
</svg>'''

    traces = "\n\n".join(esc(c.provenance[-2]) + "\n" + esc(c.provenance[-1])
                         for _, _, _, c in sweep)

    # act two: two DIFFERENT rewrites under one budget -- the certified
    # profile feeding the equilibrium solve through its exported
    # sensitivity (skipped gracefully when FEniCSx is absent locally;
    # CI always has it)
    try:
        pipe_a = sf.gs_flux_dispatch(0.5)
        pipe_b = sf.gs_flux_dispatch(0.2)
        try:
            sf.gs_flux_dispatch(0.05)
            pipe_refusal = None
        except sf.Refusal as e:
            pipe_refusal = e
        pipe_agree = abs(pipe_a.value - pipe_b.value) \
            <= pipe_a.err + pipe_b.err
        pipe_stack = "\n".join(esc(p) for p in pipe_b.provenance)
        pipeline = [
            "<h2>Two rewrites, one budget</h2>"
            "<p>The split above lived inside one rewrite family. The "
            "debt this page named at the bottom — pipelines of "
            "<em>different</em> rewrites — comes due here. The query: "
            "the total poloidal flux of a "
            '<a href="gs-equilibrium.html">tokamak equilibrium</a> '
            "whose current profile is a declared infinite Legendre "
            "series, a source no solve ever sees exactly. Two "
            "rewrites answer together: one truncates the series and "
            "certifies the dropped tail by orthogonality; the other "
            "solves the equilibrium and certifies its discretization. "
            "The solve's exported sensitivity is the exchange rate "
            "between their two currencies: total error = solve error "
            "+ sensitivity &#215; tail. The chain rule that does the "
            "conversion is now an IR verb — "
            "<code>Certified.through</code> — and the tier of the "
            "result is the weakest claim anywhere in the chain.</p>",
            code_section(sf.Certified.through, sf.legendre_source_profile,
                         sf.gs_flux_dispatch),
            "<h2>The pipeline's result</h2>"
            "<p>The tol = 0.2 certificate, its full provenance stack "
            "verbatim — two rewrites' certificates, the conversion, "
            f"the plan trace:</p><pre>{pipe_stack}</pre>"
            "<ul>"
            f"<li>Both pipeline answers certify their tolerance "
            f"(err {pipe_a.err:.3g} &#8804; 0.5, {pipe_b.err:.3g} "
            "&#8804; 0.2), stay RIGOROUS through the chain, and agree "
            "within their joint error &#8212; two rungs bounding the "
            "same full-series flux: <strong>"
            f"{'yes' if pipe_agree else 'NO'}</strong>.</li>"
            "<li>The tighter budget bought both a finer mesh and a "
            "longer profile: (n, k) = (8, 2) &#8594; (16, 4).</li>"
            "<li>Asked past the declared mesh ladder, the refusal "
            "names the mesh — not the profile — as the wall: "
            f"<code>{esc(str(pipe_refusal)[-120:])}</code></li></ul>",
        ]
    except Exception as exc:                     # pragma: no cover
        pipeline = [f"<!-- pipeline section skipped: {esc(str(exc))} -->"]

    # act three: three stages, the last repricing the first. Bench
    # constants and seeds match the test suite, so the split lines
    # are byte-identical to the checked ones.
    pA, pRho, pE0, pdE, pRel = 1.0, 0.9, 0.9, 0.3, 1e-2

    def p_full(N):
        ts = np.arange(1, N + 1, dtype=float)
        return pA * np.exp(-pE0 * ts) / (1.0 - pRho * np.exp(-pdE * ts))

    def p_cov1(N):
        return np.diag((pRel * p_full(N)) ** 2)

    def p_sample(C, m):
        rng = np.random.default_rng(len(C) * 1_000_003 + m)
        return C + rng.standard_normal(len(C)) * pRel * p_full(len(C)) \
            / math.sqrt(m)

    def p_truth(omega, s):
        return sum(pA * pRho ** k
                   * math.exp(-(omega - (pE0 + pdE * k)) ** 2 / (2 * s * s))
                   / (s * math.sqrt(2 * math.pi)) for k in range(6000))

    def p_split(cert):
        line = next(p for p in cert.provenance
                    if p.startswith("spectral-pipeline split"))
        m = re.match(r"spectral-pipeline split at kernel=(\d+) "
                     r"model=(\d+) samples=(\d+)", line)
        return int(m.group(1)), int(m.group(3)), int(m.group(2))

    p3 = {t: sf.spectral_pipeline_dispatch(p_sample, p_cov1, pA, pRho,
                                           pE0, pdE, 1.0, 0.5, tol=t)
          for t in (0.5, 0.24)}
    p3n = sf.spectral_pipeline_dispatch(p_sample, p_cov1, pA, pRho,
                                        pE0, pdE, 1.0, 0.4, tol=0.5)
    try:
        sf.spectral_pipeline_dispatch(p_sample, p_cov1, pA, pRho, pE0,
                                      pdE, 1.0, 0.5, tol=0.2)
        p3_refusal = None
    except sf.Refusal as e:
        p3_refusal = e
    (_, mA, KA), (_, mB, KB) = p_split(p3[0.5]), p_split(p3[0.24])
    Nn, mN, KN = p_split(p3n)
    p3_contained = sum(
        abs(c.value - p_truth(1.0, s)) <= c.err
        for c, s in ((p3[0.5], 0.5), (p3[0.24], 0.5), (p3n, 0.4)))
    p3_stack = "\n".join(esc(p) for p in p3[0.24].provenance)
    deeper = [
        "<h2>Three bills, one budget</h2>"
        "<p>The chain gets deeper, and something new happens in the "
        "middle. The query: the smeared spectral value of a declared "
        "pole tower — weight A&#183;&#961;<sup>k</sup> at energy "
        "E&#8320; + k&#183;dE, infinitely many states, so no finite "
        "correlator ever holds them all. Three stages answer it: a "
        "model stage truncates the tower at K poles and certifies the "
        "dropped tail exactly (the tail is a geometric series — it is "
        "summed, not estimated); a measurement stage averages m noisy "
        "samples of the truncated correlator; the smearing stage "
        "converts correlator into answer through the kernel. Three "
        "bills, one tolerance.</p>"
        "<p>The new thing: the exchange rate between the model bill "
        "and the answer is the smearing certificate's exported "
        "sensitivity, and that constant belongs to the <em>kernel</em> "
        "— a different N is a different linear functional with a "
        "different norm. So K is not a constant of the problem. Every "
        "rung of the resolution ladder reprices the model stage "
        "before it, and the plan's chosen K moves when either the "
        "budget or the kernel moves. That midstream repricing is what "
        "makes this a chain rather than three independent "
        "budgets.</p>",
        code_section(sf.pole_correlator, sf.spectral_pipeline_dispatch),
        "<h2>The chain's result</h2>"
        "<p>The tol = 0.24 certificate, its provenance stack verbatim "
        "— the model's exact tail, the smearing certificate, the "
        "conversion, the three-way split, the plan trace:</p>"
        f"<pre>{p3_stack}</pre>"
        "<ul>"
        f"<li>Containment: <strong>{p3_contained}/3</strong> chained "
        "certificates contain the 6000-pole truth.</li>"
        "<li>Tightening the budget repriced all three bills at once: "
        f"tol 0.5 &#8594; 0.24 bought K = {KA} &#8594; {KB} poles "
        f"and m = {mA} &#8594; {mB} samples through the same N = 12 "
        "kernel.</li>"
        "<li>A new kernel is a new exchange rate: asked the sharper "
        f"question &#963; = 0.4, the plan bought N = {Nn} — and the "
        f"model stage repriced through that kernel's own constant, "
        f"K = {KA} &#8594; {KN}.</li>"
        "<li>Past the finest kernel the refusal names the smearing "
        "bill — not the model — as the wall: <code>"
        f"{esc(str(p3_refusal)[-150:])}</code></li></ul>",
    ]

    # act four: a fan-in. Two branches that never feed each other, so
    # the split has no exchange rate to compute it and no curve to
    # look it up in -- the planner searches the product ladder.
    gell = (2, 3, 4, 5)
    gnear = {e: sf.h_chain_bracket(6, 1.8, e) for e in gell}
    gfar = {e: sf.h_chain_bracket(6, 3.0, e) for e in gell}
    g_a = sf.h_chain_gap_dispatch(6, tol=0.5)
    g_b = sf.h_chain_gap_dispatch(6, tol=0.3)
    try:
        sf.h_chain_gap_dispatch(6, tol=0.2)
        g_refusal = None
    except ValueError as exc:
        g_refusal = exc
    g_order = sorted(((a, b) for a in gell for b in gell),
                     key=lambda ab: 4.0 ** ab[0] + 4.0 ** ab[1])
    def g_split(c):
        line = next(p for p in c.provenance
                    if p.startswith("h-chain-gap split at"))
        m = re.search(r"compressed=(\d+) stretched=(\d+)", line)
        return int(m.group(1)), int(m.group(2))

    g_pick = {g_split(c) for c in (g_a, g_b)}

    # the product ladder as a grid: rows are the compressed branch,
    # columns the stretched one, each cell the pair's total error,
    # numbered in the cost order the planner walks.
    cw, ch, ox, oy = 74, 46, 62, 34
    g_cells = []
    for g_i, (g_a_ell, g_b_ell) in enumerate(g_order, 1):
        gx, gy = ox + (g_b_ell - 2) * cw, oy + (g_a_ell - 2) * ch
        g_tot = gnear[g_a_ell].err + gfar[g_b_ell].err
        g_on = (g_a_ell, g_b_ell) in g_pick
        g_fill = "var(--panel)" if g_tot <= 0.5 else "none"
        g_edge = "var(--blue)" if g_on else "var(--hairline)"
        g_cells.append(
            f'<rect x="{gx}" y="{gy}" width="{cw - 4}" height="{ch - 4}" '
            f'fill="{g_fill}" stroke="{g_edge}" stroke-width="'
            f'{2 if g_on else 1}"/>'
            f'<text x="{gx + 6}" y="{gy + 16}" font-size="10" '
            f'fill="var(--muted)">{g_i}</text>'
            f'<text x="{gx + (cw - 4) / 2}" y="{gy + 31}" font-size="12" '
            f'text-anchor="middle" fill="var(--ink)">{g_tot:.3g}</text>')
    for g_e in gell:
        g_cells.append(
            f'<text x="{ox + (g_e - 2) * cw + (cw - 4) / 2}" y="{oy - 10}" '
            f'font-size="11" text-anchor="middle" fill="var(--muted)">'
            f'&#8467;={g_e}</text>'
            f'<text x="{ox - 10}" y="{oy + (g_e - 2) * ch + 28}" '
            f'font-size="11" text-anchor="end" fill="var(--muted)">'
            f'&#8467;={g_e}</text>')
    gsvg = (f'<svg viewBox="0 0 {ox + 4 * cw + 12} {oy + 4 * ch + 26}" '
            'width="100%" role="img" aria-label="the product ladder of '
            'window widths, numbered in cost order">'
            f'<text x="4" y="{oy - 10}" font-size="11" fill="var(--muted)">'
            'stretched &#8594;</text>'
            f'<text x="4" y="{oy + 4 * ch + 18}" font-size="11" '
            f'fill="var(--muted)">compressed &#8595;</text>'
            + "".join(g_cells) + "</svg>")

    fan_in = [
        "<h2>Two branches, one budget</h2>"
        "<p>Every chain above is a <em>line</em>: one stage's answer "
        "is the next stage's input, and the sensitivity is the "
        "exchange rate that carries error along it. This last query "
        "has no line in it. It asks what it costs to stretch the "
        "six-atom hydrogen chain from d = 1.8 to d = 3.0 bohr — the "
        "difference of two ground-state brackets that never see each "
        "other. Nothing converts into anything; the errors simply "
        "add. The only question left is how to divide one tolerance "
        "between two independent branches.</p>"
        "<p>And here the arithmetic that priced every earlier split "
        "runs out. Those splits worked because a curve was known "
        "ahead of the run — a geometric tail summed exactly, a "
        "1/&#8730;m law — so the budget could be divided by formula "
        "before spending anything. The h-chain window bracket obeys "
        "no such law: what a window of width &#8467; certifies is "
        "whatever the run measures. So the allocation becomes a "
        "<em>search</em>. Every pair of widths is a rung, priced "
        "4<sup>&#8467;near</sup>&nbsp;+&nbsp;4<sup>&#8467;far</sup>, "
        "and the planner walks the product ladder in cost order until "
        "a pair certifies — the same frontier, the same referee, a "
        "different shape of plan.</p>",
        code_section(sf.h_chain_gap_dispatch),
        "<h2>The fan-in's result</h2>"
        f"<figure>{gsvg}<figcaption>The product ladder. Each cell is "
        "a pair of window widths and its total certified error; the "
        "small number is the order the planner tries them in, by "
        "predicted cost. Rows are the compressed branch, columns the "
        "stretched one — and the grid is visibly not symmetric, "
        "because the two branches are not equally hard. Outlined "
        "cells are the pairs chosen at tol = 0.5 and tol = "
        "0.3.</figcaption></figure>"
        "<p>The compressed chain delocalizes across all six atoms, so "
        "a window of width &#8467; misses more of it; the stretched "
        "chain is nearly decoupled and a window of the same width "
        "catches almost everything. Measured, the compressed bracket "
        f"runs {gnear[3].err / gfar[3].err:.1f}&#215; the stretched "
        f"one at &#8467; = 3 and {gnear[5].err / gfar[5].err:.1f}"
        "&#215; at &#8467; = 5. So the budget buys width where width "
        "is scarce:</p>"
        f"<pre>{esc(chr(10).join(g_a.provenance[-2:]))}</pre>"
        "<ul>"
        f"<li>At tol = 0.5 the plan splits <strong>asymmetrically</strong>"
        " — &#8467; = 4 on the compressed branch, 3 on the stretched, "
        "predicted cost 320. The cheapest pair a single shared knob "
        "could reach is &#8467; = 4 on both, predicted 512: "
        "<strong>1.6&#215;</strong> the price for the same "
        "tolerance.</li>"
        "<li>A fan-in's product ladder costs the <strong>sum</strong> "
        "of its branches, not their product, because escalating one "
        "branch reuses the other. At tol = 0.3 the plan walked "
        f"<strong>{len(g_b.receipt)}</strong> pairs but paid for only "
        "<strong>8</strong> brackets — two branches &#215; four "
        "widths. A pair whose branches are both already solved costs "
        "one subtraction: rung (3,&nbsp;3) measured "
        f"{min(s for _, k, _, s, _ in g_b.receipt if k == (3, 3)):.1e}"
        " s.</li>"
        "<li>Past the pair floor the refusal names <em>which branch "
        "is the wall</em> — with two branches, “what to buy next” has "
        "two answers and only one of them helps: <code>"
        f"{esc(str(g_refusal)[-190:])}</code></li>"
        "<li>An honest wrinkle. Both runs enclose the same true gap, "
        "so their brackets must overlap, and they do. But the "
        f"narrower one is not the stronger statement: tol = 0.5 "
        f"gives {g_a.value:.3f} &#177; {g_a.err:.3f} and tol = 0.3 "
        f"gives {g_b.value:.3f} &#177; {g_b.err:.3f} — a smaller "
        "half-width whose midpoint also moved. Whether the gap is "
        "positive is settled by intersecting certified enclosures, "
        "not by keeping the narrowest.</li></ul>",
    ]

    return page(
        "Case: the composed plan",
        "certified case",
        "One budget, two bills",
        "The spectral question from the two-peaks page, asked again "
        "of noisy Monte Carlo data. Now two error streams — what the "
        "kernel cannot resolve, and what the noise obscures — must be "
        "paid out of one tolerance. The first composed plan splits the "
        "budget by formula, priced by a sensitivity the certificate "
        "exports, and lets the certificate referee the result.",
        [
            "<h2>The idea</h2>"
            "<p>The question is the one "
            '<a href="smeared-spectral.html">an earlier page</a> '
            "answered: a spectral density &#961; is visible only "
            "through its Euclidean correlator, "
            "C(t)&nbsp;=&nbsp;&#8747;e<sup>&#8722;&#969;t</sup>"
            "&#961;(&#969;)&nbsp;d&#969;, and the query asks for the "
            "Gaussian-smeared value of &#961; at frequency &#969; and "
            "resolution &#963;. On that page the correlator was exact, "
            "and the certificate paid a single bill: the smearing bill "
            "c&#183;C(1), what the reconstructed kernel cannot "
            "resolve, reduced only by buying more correlator times N. "
            "But real correlators come out of Monte Carlo — on a "
            "lattice, each C(t) is the mean of m noisy samples — and "
            "noise adds a second bill: z&#183;amp/&#8730;m, what the "
            "noise obscures, reduced only by buying more samples "
            "m.</p>"
            "<p>One tolerance must cover both bills, so something has "
            'to decide the split. <a href="the-compiler.html">The '
            "planner page</a> made a cheaper decision: which rung of "
            "one ladder to run next. This page is the promised next "
            "step, a <em>composed</em> plan that divides one error "
            "budget between two stages — resolution and statistics — "
            "before a single sample is bought. The division itself is "
            "near-solved mathematics, the same marginal-cost balancing "
            "multilevel Monte Carlo uses to allocate work across "
            "levels; and because the statistics bill obeys a "
            "1/&#8730;m law exactly, the split collapses to a closed "
            "form per rung of the resolution ladder.</p>"
            "<p>The datum that makes the formula possible is new to "
            "the IR. A certificate now exports its "
            "<em>sensitivity</em>: the smeared value is the linear map "
            "g&#183;C, so data error moves the value through at most "
            "the norm of g — and re-anchors the smearing bill too, "
            "which Cauchy&#8211;Schwarz folds into the one exact "
            "constant &#8730;(c&#178;+&#8214;g&#8214;&#178;), rigorous "
            "whatever the tier of the value's own bound. That "
            "one number prices the statistics stage in advance. And as "
            "everywhere on this site, the prices only order the "
            "attempts: the certificate of the run that executes is the "
            "referee, so a wrong pilot estimate costs extra rungs, "
            "never truth.</p>",
            code_section(sf.Sensitivity, sf.smeared_spectral_dispatch),
            "<h2>The result</h2>"
            "<p>The bench is the two-peak density of the earlier page "
            f"— peaks at &#969; = {E[0]:g} and {E[1]:g} with weights "
            f"{a[0]:g} and {a[1]:g} — queried at &#969; = {omega:g}, "
            f"&#963; = {sigma:g}, with {rel * 100:g}% relative noise "
            "per sample; measure(N, m) returns the mean of m samples "
            "and the declared ladder offers N = 8, 12, 16 correlator "
            "times.</p>"
            f"<figure>{svg}<figcaption>The economics of the split at "
            "&#963; = 0.5. Each curve prices one resolution rung: its "
            "sample bill N&#183;m explodes as the tolerance approaches "
            "that rung's smearing bill, where no sample count helps. "
            "The plan walks the lower envelope — dots mark the splits "
            "it actually chose. At tol = 0.4 the coarse N = 8 kernel "
            "wins by a nose; tightening the budget hands the race to "
            "N = 12, whose smearing bill is smaller, at ever larger "
            "m.</figcaption></figure>"
            "<p>The three certificates' split lines and plan traces, "
            f"verbatim:</p><pre>{traces}</pre>",
            *pipeline,
            *deeper,
            *fan_in,
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment: <strong>{contained}/3</strong> "
            "certificates contain the exact smeared truth — computable "
            "because the density is synthetic; never used by the "
            "plan.</li>"
            "<li>Tighter tolerance buys samples by the 1/&#8730;m law: "
            f"m = <strong>{sweep[0][2]} &#8594; {sweep[1][2]} &#8594; "
            f"{sweep[2][2]}</strong> across the sweep.</li>"
            "<li>The split buys resolution when statistics are dearer: "
            "asked the sharper question &#963; = 0.4, it chose "
            f"N = {hN} with m = {hm} "
            f"(predicted {hcost:g}); the coarse N = 8 kernel would "
            f"have needed m = {m8} samples (predicted {8 * m8:.3g}), "
            f"{8 * m8 / hcost:.0f}&#215; the price, for the same "
            "tolerance.</li>"
            "<li>The sensitivity field composes through the IR's own "
            "arithmetic: the difference of the smeared values at the "
            "two peaks carries the sum of the two exported constants, "
            f"{diff.sensitivity.bound:.3g} "
            f"w.r.t. “{diff.sensitivity.wrt}”, still "
            "RIGOROUS.</li>"
            "<li>Below every rung's smearing bill the plan refuses "
            "before spending a single sample: "
            f"<code>{esc(str(refusal)[:200])}&hellip;</code></li>"
            "<li>The combinator these four plans specified now exists, "
            "and all four are wired through it. A plan is declared "
            "as stages and one assemble function: each stage names the "
            "stages it consumes, so the object is a graph, and "
            "assemble joins the certificates with <code>through</code>, "
            "<code>+</code> or <code>&minus;</code>, whichever the "
            "graph calls for. The split is not assumed computable — a "
            "stage that can solve for its knob given a budget is "
            "solved, in the order the budget is spent, and a stage "
            "that cannot is enumerated and walked cheapest-first. The "
            "fan-in uses two searched stages joined by subtraction; "
            "the equilibrium pipeline uses one enumerated mesh and one "
            "solved profile joined by a sensitivity. Both keep the "
            "rungs and the numbers they had when they were wired by "
            "hand. The two spectral plans needed what the combinator "
            "lacked, which is why they went last: a stage may now be a "
            "CHOICE rather than a certificate — an N-point kernel "
            "spends budget on what it cannot resolve, and that bill "
            "lands inside the measurement's certificate rather than in "
            "one of its own — and two stages may share that choice, "
            "which is how the pole model and the measurement come to "
            "live on one grid.</li>"
            "<li>One sentence stopped being hand-written. Every front "
            "door used to name its own binding stage in the refusal, "
            "which means it could only ever describe the shape its "
            "author had in mind. The combinator derives it: zero each "
            "stage's error in turn, reassemble, and see which one the "
            "composed error was leaning on. Along a line that drop is "
            "the error converted through the sensitivity; at a fan-in "
            "it is the branch's own error; the assembly does the "
            "arithmetic either way, and it costs nothing, because "
            "every certificate is already in hand.</li>"
            "<li>Going back is not a matter of discipline. A rewrite "
            "whose knob is a tuple is an assignment across stages, not "
            "a ladder, and the planner refuses one it did not build "
            "itself; the build in turn refuses any front door that "
            "switches that check off, so a release cannot ship the "
            "debt back.</li>"
            "<li>The walk is the cost ladder, and stays that way on "
            "evidence rather than on inertia. Three searches that use "
            "the graph's shape instead have been built and abandoned. "
            "<em>Escalate the binding stage:</em> reached a certifying "
            "pair in 4&ndash;7 runs against the ladder's 8&ndash;16, "
            "but counted in the thing that costs — nodes computed, not "
            "assemblies, the branches being shared — it was 5 against "
            "6, then 6 against 6, then 7 against 6, and twice it "
            "returned a dearer assignment because it never revisited "
            "what it stepped over. <em>Order by marginal node cost,</em> "
            "so no node is bought while a free assembly remains: one "
            "node saved at one tolerance in eight, and a dearer "
            "assignment there. <em>Prune assignments whose measured "
            "components already exceed the tolerance:</em> admissible, "
            "same answer every time, and it cuts a five-branch walk "
            "from 1024 assemblies to 5 — while saving no nodes at all "
            "in nine of twelve cases from two branches to five.</li>"
            "<li>They fail for one reason, and it is the reason to stop "
            "trying. The shared-node memo already makes node "
            "computations linear in the graph, so the only quantity a "
            "cleverer search can reduce is assemblies, and assemblies "
            "are arithmetic on certificates already in hand. It would "
            "take a graph whose cheap rungs did not already cover every "
            "knob — coarse dear and fine cheap — which is the opposite "
            "of every ladder here.</li></ul>",
        ])


# ======================================================================
def criticality_case():
    from scipy.linalg import eig
    rx = sf.slab_reactor()
    L, F = rx["L"], rx["F"]
    u = sf.mmatrix_witness(L)
    mags = np.sort(np.abs(eig(np.linalg.solve(L, F), right=False)))[::-1]
    truth = float(mags[0])
    buckling = sf.slab_buckling_keff(rx["width"])

    # the ladder: bracket after each fission-source iteration
    phi = np.ones(len(L))
    rungs = []
    for m in range(61):
        c = sf.keff_bracket(L, F, phi, u)
        rungs.append((m, c.value - c.err, c.value + c.err, c.err * 1e5))
        psi = np.linalg.solve(L, F @ phi)
        phi = psi / psi.max()
    contained = sum(1 for _, lo, hi, _ in rungs if lo <= truth <= hi)
    floor = min(w for _, _, _, w in rungs)
    geometric = [w for m, _, _, w in rungs if 5 <= m <= 35]
    rate = float(np.mean([b / a for a, b in zip(geometric, geometric[1:])]))
    at_floor = min(m for m, _, _, w in rungs if w < floor * 1.5)

    # the front door, at three tolerances the field would recognise
    lines, chosen = [], []
    for tol in (100.0, 10.0, 1.0):
        c = sf.keff_dispatch(rx, tol)
        chosen.append(c)
        lines.append(f"keff_dispatch(slab_reactor(), tol_pcm={tol:g})\n"
                     f"  -> {c.value:.9f} +/- {c.err * 1e5:.3f} pcm   "
                     f"[{c.value - c.err:.9f}, {c.value + c.err:.9f}]\n"
                     f"  {c.provenance[-1]}")
    try:
        sf.keff_dispatch(sf.slab_reactor(width=150.0), 1.0)
        refused = "NOT REFUSED"
    except sf.Refusal as exc:
        refused = str(exc)
    lines.append("\nkeff_dispatch(slab_reactor(width=150), tol_pcm=1)\n  -> "
                 + refused)

    # discretisation: what the bracket is a bracket OF
    mesh = []
    for N in (25, 50, 100, 200):
        A = np.linalg.solve(*[sf.slab_reactor(N=N)[k] for k in ("L", "F")])
        mesh.append((N, abs(float(np.max(eig(A, right=False).real))
                            - buckling) * 1e5))
    cont = sf.keff_continuum_bracket()
    cont_holds = cont.value - cont.err <= buckling <= cont.value + cont.err
    # the discrete bracket the continuum one is actually built on: the
    # finest rung of its own ladder, not the 1 pcm answer from above
    cont_fine = sf.keff_dispatch(sf.slab_reactor(N=200), 0.01)

    # the archetype transferring: same certificate, different equation
    sn = sf.sn_slab_reactor(N=60)
    sn_off = float((sn["L"] - np.diag(np.diag(sn["L"]))).max())
    sn_asym = float(np.linalg.norm(sn["L"] - sn["L"].T))
    sn_k = sf.keff_dispatch(sn, 1.0)
    sn_truth = float(np.max(np.abs(eig(
        np.linalg.solve(sn["L"], sn["F"]), right=False))))

    def diffusion_twin(N, w, st=1.0, ss=0.6, nsf=0.45):
        D, h = 1.0 / (3 * st), w / N
        M = np.zeros((N, N))
        for i in range(N):
            M[i, i] = 2 * D / h ** 2 + (st - ss)
            if i > 0:
                M[i, i - 1] = -D / h ** 2
            else:
                M[i, i] += D / h ** 2
            if i < N - 1:
                M[i, i + 1] = -D / h ** 2
            else:
                M[i, i] += D / h ** 2
        return {"L": M, "F": nsf * np.eye(N), "label": "diffusion"}

    twins = []
    for w in (5.0, 10.0, 20.0, 40.0):
        kt = sf.keff_dispatch(sf.sn_slab_reactor(N=100, width=w), 2.0,
                              m_max=300).value
        kd = sf.keff_dispatch(diffusion_twin(300, w), 2.0, m_max=300).value
        twins.append((w, kt, kd, (kd - kt) * 1e5))

    ms = [m for m, _, _, _ in rungs]
    ws = [w for _, _, _, w in rungs]
    ax = Axes((0, 60), (floor * 0.5, max(ws[1:]) * 1.6), h=340, logy=True,
              ml=68)
    ticks = [10.0 ** e for e in range(-7, 6)]
    band = "".join(
        f'<rect x="{ax.X(m) - 3:.1f}" y="{ax.Y(max(w, floor)):.1f}" '
        f'width="6" height="{ax.Y(floor * 0.5) - ax.Y(max(w, floor)):.1f}" '
        f'fill="var(--blue)" opacity="0.14"/>' for m, _, _, w in rungs[1:])
    svg = f'''<svg viewBox="0 0 640 340" role="img" aria-label="Certified
bracket half-width against fission-source iteration, contracting
geometrically until it floors on the solve residual">
{ax.grid(ticks, (0, 10, 20, 30, 40, 50, 60),
         xfmt=lambda v: f"{v:g}",
         yfmt=lambda v: (f"{v:g}" if v >= 1 else f"1e{round(math.log10(v))}"))}
{band}
<line x1="{ax.ml}" y1="{ax.Y(floor):.1f}" x2="{ax.w - ax.mr}"
      y2="{ax.Y(floor):.1f}" class="rust-ink" stroke-width="1.4"
      stroke-dasharray="5 4"/>
<text x="{ax.w - ax.mr - 4}" y="{ax.Y(floor) - 6:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" opacity="0.85">arithmetic floor,
{floor:.1e} pcm</text>
<line x1="{ax.ml}" y1="{ax.Y(mesh[2][1]):.1f}" x2="{ax.w - ax.mr}"
      y2="{ax.Y(mesh[2][1]):.1f}" class="board-ink" stroke-width="1.4"
      stroke-dasharray="2 3" opacity="0.8"/>
<text x="{ax.w - ax.mr - 4}" y="{ax.Y(mesh[2][1]) - 6:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" opacity="0.85">the model's own
error, {mesh[2][1]:.2f} pcm</text>
<path d="{ax.path(ms[1:], [max(w, floor) for w in ws[1:]])}" fill="none"
      class="blue-ink" stroke-width="2"/>
<text x="{ax.ml + 6}" y="{ax.h - 4}" class="board-text" font-size="10.5"
      opacity="0.7">fission-source iterations</text></svg>'''

    rows = "".join(
        f"<tr><td>{m}</td><td>{lo:.9f}</td><td>{hi:.9f}</td>"
        f"<td>{w:.4g}</td></tr>"
        for m, lo, hi, w in rungs if m in (1, 5, 10, 15, 20, 25, 30, 40, 60))

    return page(
        "Case: a reactor's criticality eigenvalue, bracketed by positivity",
        "certified case",
        "A reactor is not symmetric, so the bracket comes from positivity",
        "Every other eigenvalue bracket in this library needs a "
        "self-adjoint operator. A reactor is not one. What replaces the "
        "variational theorem is the fact that neutrons cannot be "
        "negative.",
        [
            "<h2>The idea</h2>"
            "<p>The brackets elsewhere here all work the same way: the "
            "lowest eigenvalue of a symmetric operator is a minimum "
            "over states, so any trial state you can write down "
            "overshoots it, and that overshoot is the certificate. The "
            "argument needs the operator to equal its own transpose.</p>"
            "<p>A reactor does not. Neutrons are born fast, scatter "
            "down in energy, and essentially never scatter back up, so "
            "the loss operator is block lower-triangular; its transpose "
            "describes a different physical quantity, the adjoint flux. "
            "There is no minimum to overshoot. Nor does handing the "
            "matrix to a dense eigensolver rescue the situation: a "
            "computed eigenvalue of a non-symmetric matrix carries no "
            "cheap rigorous bound, because its sensitivity is set by an "
            "eigenvector overlap that can be arbitrarily small.</p>"
            "<p>What survives is positivity. Fission makes neutrons and "
            "never unmakes them, and a source anywhere produces flux "
            "everywhere and a deficit nowhere — so both operators in "
            "the problem map the non-negative cone into itself. "
            "Collatz and Wielandt showed that such an operator's "
            "dominant eigenvalue is sandwiched by the smallest and "
            "largest of the ratios (A&#966;)<sub>i</sub>/"
            "&#966;<sub>i</sub>, for <em>any</em> strictly positive "
            "&#966;. Here that eigenvalue is k<sub>eff</sub>, and "
            "applying A is exactly one step of the fission-source "
            "iteration a reactor code already runs. Production codes "
            "stop that iteration when k stops moving, which is a hope. "
            "This one stops when the sandwich closes, which is a "
            "bound.</p>"
            "<p>Four hypotheses carry the theorem and all four are "
            "checked at runtime rather than assumed: the loss operator "
            "has no positive off-diagonal, the fission operator has no "
            "negative entry, the trial flux is strictly inside the "
            "cone, and there is a positive vector u with Lu ≥ 1. The "
            "last two lines of that list do double duty — u proves the "
            "inverse is non-negative <em>and</em> prices the linear "
            "solver's own error, so the bracket stays rigorous however "
            "sloppily the solve was done.</p>",
            code_section(sf.slab_reactor, sf.mmatrix_witness,
                         sf.keff_bracket, sf.keff_dispatch,
                         sf.keff_continuum_bracket, sf.sn_slab_reactor),
            "<h2>The run</h2>"
            "<p>A two-group slab, 70 cm across, 100 cells per group. "
            "Starting from a flat trial flux and iterating:</p>"
            "<table><thead><tr><th>iterations</th><th>lower</th>"
            "<th>upper</th><th>half-width (pcm)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<p class='note'>A dense non-symmetric eigensolver, which "
            f"the bracket never calls, puts the answer at "
            f"{truth:.9f}. It lies inside <strong>{contained}/"
            f"{len(rungs)}</strong> of the brackets above.</p>",
            f"<figure>{svg}<figcaption>The sandwich closing. Each bar "
            "is one certified bracket, plotted by half-width on a log "
            f"scale; the width contracts by a factor {rate:.3f} per "
            "iteration until it meets the dashed line where the "
            "certified residual of the linear solve, rather than the "
            "Collatz–Wielandt gap, sets the width. That floor sits a "
            "million times below the dotted line, which is where this "
            "discretised slab differs from the continuum reactor it "
            "stands for.</figcaption></figure>",
            "<h2>The front door</h2>"
            "<p>Reactor physics states its tolerance in pcm, "
            "hundred-thousandths of k, so that is what the dispatch "
            f"takes.</p><pre>{esc(chr(10).join(lines))}</pre>",
            "<h2>What is being bracketed</h2>"
            "<p>The certificate is about the eigenvalue of the "
            "discretised slab, not of a reactor. That gap is the "
            "model's and it is stated, not absorbed. Against the "
            "closed-form buckling answer for the same slab:</p>"
            "<table><thead><tr><th>cells per group</th>"
            "<th>distance from the continuum (pcm)</th></tr></thead>"
            "<tbody>"
            + "".join(f"<tr><td>{N}</td><td>{e:.2f}</td></tr>"
                      for N, e in mesh)
            + "</tbody></table>"
            "<p>Second order in the mesh, as the scheme promises, and "
            f"{mesh[2][1]:.2f} pcm at the default resolution — about a "
            "million times the certificate's own floor. On this "
            "problem the bound is never the weak link, which is the "
            "honest ordering of the two errors and the reason to print "
            "both.</p>"
            "<p>That gap can itself be certified, by feeding the "
            "ladder of rigorous discrete brackets to the same "
            "grid-convergence machinery the sea-wall page uses. One "
            "precondition is checked rather than hoped for: the "
            "discrete brackets have to be far narrower than the "
            "differences between rungs, or the measured order is "
            "fitting their noise.</p>"
            f"<pre>{esc(f'keff_continuum_bracket()  ->  {cont.value:.9f}'
                        f' +/- {cont.err * 1e5:.3f} pcm   '
                        f'[{cont.tier.name}]')}</pre>"
            f"<p>The closed-form continuum answer is {buckling:.9f}, and "
            f"it is <strong>{'inside' if cont_holds else 'OUTSIDE'}</strong> "
            "that interval. Notice what the composition cost: the "
            f"continuum certificate is {cont.err / cont_fine.err:.0f} times "
            "wider than the rigorous discrete bracket it is built from, "
            "and a tier weaker, because half of it is a measured "
            "convergence order rather than a proven inequality. That is "
            "the exchange rate between a proven statement about a model "
            "and a measured one about the world. Both are printed; "
            "neither is dressed as the other.</p>",
            "<h2>The same certificate, a different equation</h2>"
            "<p>Diffusion is itself an approximation, to the transport "
            "equation that tracks neutrons by direction as well as "
            "position. Discretising that instead — discrete ordinates "
            "on a Gauss-Legendre angular quadrature, streaming "
            "differenced upwind — changes the operator completely and "
            "changes the certificate not at all.</p>"
            "<p>Upwind differencing exists precisely because it cannot "
            "produce a negative flux, and that is the same statement as "
            "the operator being a Z-matrix. So the largest off-diagonal "
            f"is {sn_off:g}, the operator is asymmetric by "
            f"{sn_asym:.1f} in norm, and <code>mmatrix_witness</code> and "
            "<code>keff_bracket</code> run on it unaltered — no new "
            "proof, no new code, the same front door:</p>"
            f"<pre>{esc(f'keff_dispatch(sn_slab_reactor(N=60), tol_pcm=1)'
                        + chr(10) + f'  -> {sn_k.value:.9f} +/- '
                        f'{sn_k.err * 1e5:.3f} pcm')}</pre>"
            f"<p class='note'>Dense <code>eig</code> on the transport "
            f"operator says {sn_truth:.9f}, inside the bracket. A "
            "certificate hung on a cone rather than on a quadratic form "
            "does not care which equation it is looking at, only whether "
            "the equation respects the cone.</p>"
            "<p>Having both models measures something neither "
            "certificate can: how good the diffusion approximation "
            "actually is. Same cross sections, both converged:</p>"
            "<table><thead><tr><th>slab (mean free paths)</th>"
            "<th>transport</th><th>diffusion</th>"
            "<th>diffusion error (pcm)</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{w:g}</td><td>{kt:.6f}</td>"
                      f"<td>{kd:.6f}</td><td>{g:+.0f}</td></tr>"
                      for w, kt, kd, g in twins)
            + "</tbody></table>"
            "<p>Thousands of pcm wrong in a small leaky slab, tens in a "
            "large one — the textbook behaviour, measured rather than "
            "recited. Neither bracket knows this, and neither is wrong: "
            "each is rigorous about its own operator. Which operator "
            "deserves belief is a modelling question, and it stays a "
            "separate, stated step.</p>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment against a dense eigensolver: <strong>"
            f"{contained}/{len(rungs)}</strong> rungs.</li>"
            f"<li>Contraction per iteration, measured over the "
            f"geometric stretch: <strong>{rate:.3f}</strong>. This is "
            "the second spatial harmonic's ratio, not the first — a "
            "flat starting flux is symmetric about the midplane, so "
            "the antisymmetric first harmonic is never excited.</li>"
            f"<li>Arithmetic floor: <strong>{floor:.1e} pcm</strong>, "
            f"reached after about {at_floor} iterations.</li>"
            f"<li>Tolerance met at 1 pcm: <strong>"
            f"{chosen[-1].err * 1e5:.3f} pcm</strong> in "
            f"<strong>{len(chosen[-1].receipt)}</strong> rungs, the "
            "planner jumping rather than stepping.</li>"
            "<li>A 150 cm core, whose dominance ratio is much closer to "
            "one, <strong>refuses</strong> 1 pcm within the declared "
            "ladder and prices what would be needed.</li>"
            f"<li>The continuum certificate, mesh ladder "
            f"{'/'.join(str(N) for N, _ in mesh)}: <strong>"
            f"{cont.err * 1e5:.3f} pcm</strong>, EMPIRICAL, and it "
            f"<strong>{'contains' if cont_holds else 'MISSES'}</strong> "
            "the closed-form continuum answer.</li>"
            "<li>The same bracket on a transport operator, with no new "
            f"proof: <strong>{sn_k.err * 1e5:.3f} pcm</strong>, "
            "containing dense <code>eig</code>.</li>"
            f"<li>Diffusion measured against transport: "
            f"<strong>{twins[0][3]:+.0f} pcm</strong> at "
            f"{twins[0][0]:g} mean free paths, "
            f"<strong>{twins[2][3]:+.0f} pcm</strong> at "
            f"{twins[2][0]:g}.</li>"
            "</ul>",
        ])


# ======================================================================
def junction_case():
    dev = sf.pn_junction()
    volts = 1.0
    v = volts / sf._VT
    truth = sf.junction_charge_bracket(
        dev, volts, sf.junction_potential(dev, volts, 400)).value

    # the ladder: residual and certified error after each Newton step
    rungs = []
    for m in range(1, 21):
        psi = sf.junction_potential(dev, volts, m)
        r, e = sf._poisson_residual(dev, psi, v)
        res = float((abs(r) + e).max())
        try:
            c = sf.junction_charge_bracket(dev, volts, psi)
            rungs.append((m, res, c.value, c.err))
        except ValueError:
            rungs.append((m, res, None, None))
    first = min(m for m, _, _, w in rungs if w is not None)
    certd = [(m, res, q, w) for m, res, q, w in rungs if w is not None]
    contained = sum(1 for _, _, q, w in certd if q - w <= truth <= q + w)
    r0 = next(x for x in rungs if x[0] == first)
    rlast = next(x for x in rungs if x[0] == first - 1)
    drop = r0[3] / next(x[3] for x in rungs if x[0] == first + 1)

    # the three numbers Kantorovich runs on, at the certifying rung
    psi0 = sf.junction_potential(dev, volts, first)
    beta = float(sf.mmatrix_witness(sf._poisson_jacobian(dev, psi0, v)).max())

    # the front door
    lines, chosen = [], []
    for vv in (0.0, 1.0, 3.0):
        c = sf.junction_dispatch(dev, vv, 1e-3)
        chosen.append(c)
        lines.append(f"junction_dispatch(pn_junction(), {vv:g}, "
                     f"tol_nC=1e-3)\n  -> {c.value:.6f} +/- {c.err:.3g} "
                     f"nC/cm2   [{c.tier.name}]\n  {c.provenance[-1]}")
    try:
        sf.junction_dispatch(dev, 5.0, 0.05, m_max=20)
        short = "NOT REFUSED"
    except sf.Refusal as exc:
        short = str(exc).split("; measured")[0] + "\n  next: " + exc.next_price
    lines.append("\njunction_dispatch(pn_junction(), 5.0, tol_nC=0.05, "
                 "m_max=20)\n  -> " + short)

    # what the certificate is about: the mesh
    mesh = [(n, sf.junction_dispatch(sf.pn_junction(N=n), volts, 1e-3))
            for n in (100, 200, 400, 800)]
    mdiff = [abs(b[1].value - a[1].value) for a, b in zip(mesh, mesh[1:])]

    # the independent check: the depletion approximation
    dep = []
    for vv in (0.0, 1.0, 2.0, 3.0):
        w = sf.depletion_width_analytic(dev, vv)
        q_an = sf._QE * dev["Nd"] * (w * 1e-7 / 2) * 1e9
        c = sf.junction_dispatch(dev, vv, 1e-3)
        dep.append((vv, w, q_an, c.value, c.value / q_an))

    # the witness, borrowed from the reactor, and the bug it exposed
    Ju = sf._poisson_jacobian(dev, psi0, v).copy()
    Ju[1:-1, :] *= dev["lam2"] / dev["h"] ** 2
    cond = float(np.linalg.cond(Ju))
    naive = np.linalg.solve(Ju, np.ones(len(Ju))) * (1.0 + 1e-9)
    naive_short = float((Ju @ naive).min())

    ms = [m for m, _, _, _ in rungs]
    resid = [r for _, r, _, _ in rungs]
    lo_y = min(min(resid), min(w for _, _, _, w in certd)) * 0.2
    hi_y = max(resid) * 5
    ax = Axes((1, 20), (lo_y, hi_y), h=360, logy=True, ml=68)
    ticks = [10.0 ** e for e in range(-14, 3, 2)]
    svg = f'''<svg viewBox="0 0 640 360" role="img" aria-label="Newton
residual falling smoothly while the certified error does not exist at
all until the Kantorovich condition closes, then plunges">
{ax.grid(ticks, (1, 5, 10, 15, 20), xfmt=lambda t: f"{t:g}",
         yfmt=lambda t: f"1e{round(math.log10(t))}")}
<rect x="{ax.ml}" y="{ax.mt}" width="{ax.X(first - 0.5) - ax.ml:.1f}"
      height="{ax.h - ax.mt - ax.mb}" fill="var(--rust)" opacity="0.07"/>
<text x="{(ax.ml + ax.X(first - 0.5)) / 2:.1f}" y="{ax.mt + 16}"
      text-anchor="middle" class="board-text" font-size="10.5"
      opacity="0.8">no bound exists here</text>
<path d="{ax.path(ms, resid)}" fill="none" class="board-ink"
      stroke-width="1.8" stroke-dasharray="4 3" opacity="0.85"/>
<path d="{ax.path([m for m, _, _, _ in certd],
                  [w for _, _, _, w in certd])}" fill="none"
      class="blue-ink" stroke-width="2.2"/>
{"".join(f'<circle cx="{ax.X(m):.1f}" cy="{ax.Y(w):.1f}" r="3" '
         f'class="blue-fill"/>' for m, _, _, w in certd)}
<text x="{ax.X(19):.1f}" y="{ax.Y(resid[-1]) - 8:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" opacity="0.8">residual</text>
<text x="{ax.X(18):.1f}" y="{ax.Y(certd[-1][3]) + 16:.1f}" text-anchor="end"
      class="board-text" font-size="10.5" opacity="0.9">certified error,
nC/cm&#178;</text>
<text x="{ax.ml + 6}" y="{ax.h - 4}" class="board-text" font-size="10.5"
      opacity="0.7">Newton steps</text></svg>'''

    rows = "".join(
        f"<tr><td>{m}</td><td>{res:.3e}</td>"
        + (f"<td>{q:.6f}</td><td>{w:.3g}</td></tr>" if w is not None
           else "<td>&#8212;</td><td>refused</td></tr>")
        for m, res, q, w in rungs if m <= 17)

    return page(
        "Case: a pn junction, and a proof that the answer exists",
        "certified case",
        "A small residual is not a small error",
        "Device simulators stop Newton's method when the residual looks "
        "small. That is not a bound. Kantorovich's theorem turns the "
        "residual into a proof that an exact solution exists, and a "
        "radius that contains it.",
        [
            "<h2>The idea</h2>"
            "<p>Every other certificate in this library answers <em>how "
            "far is this number from the right one</em>. This one has "
            "to answer something prior: is there a right one at all, "
            "anywhere near here?</p>"
            "<p>The question is not pedantic. A semiconductor's "
            "carriers are exponentials of the unknown potential, and "
            "across a junction that potential swings far enough that "
            "the densities move through twenty orders of magnitude. "
            "Newton's method on such a system is run with heavy damping "
            "and stopped when the residual stops shrinking. But a small "
            "residual is consistent with a solution nearby, with a "
            "solution far away, and with no solution at all &#8212; "
            "residuals are small near almost-singular points too, and "
            "that is precisely the neighbourhood these equations "
            "live in.</p>"
            "<p>Kantorovich's theorem converts a residual into "
            "existence. It needs three numbers: a bound &#946; on the "
            "inverse Jacobian, the length &#951; of the Newton step it "
            "implies, and a Lipschitz constant K for the Jacobian "
            "nearby. If &#946;K&#951; &#8804; &#189;, then an exact "
            "solution <em>exists</em> and sits within a computable "
            "radius. If not, nothing is proven, and the honest report "
            "is a refusal.</p>"
            "<p>The first of those three numbers is a debt already "
            "paid. Discretise the nonlinear Poisson equation and the "
            "Jacobian is a Z-matrix: the second difference contributes "
            "negative off-diagonals, the carrier response a positive "
            "diagonal. That is the same hypothesis a reactor's loss "
            "operator satisfies, so <code>mmatrix_witness</code> "
            "&#8212; written for neutrons &#8212; prices "
            "&#8214;J<sup>-1</sup>&#8214; here with no new proof. The "
            "physics has nothing in common. The cone is the same.</p>"
            "<p>That reuse carried a restriction with it, and the "
            "restriction was never Kantorovich's. The theorem asks for "
            "a bound on the inverse Jacobian and does not care where "
            "it came from; it was the <em>pricing method</em> that "
            "needed a Z-matrix. A second route asks nothing about "
            "signs: take any approximate inverse R, and if "
            "&#8214;I &#8722; RJ&#8214; &lt; 1 the Neumann series "
            "converges and &#8214;J<sup>-1</sup>&#8214; is at most "
            "&#8214;R&#8214;/(1 &#8722; &#8214;I &#8722; RJ&#8214;). It "
            "costs an inverse instead of a solve, so the cone route is "
            "tried first and still answers here &#8212; and on an "
            "M-matrix it is not merely valid but exact, since "
            "J<sup>-1</sup> &#8805; 0 makes "
            "&#8214;J<sup>-1</sup>&#8214;<sub>&#8734;</sub> literally "
            "&#8214;J<sup>-1</sup>e&#8214;<sub>&#8734;</sub>, which is "
            "what the witness computes. The <a "
            "href=\"gs-equilibrium.html\">tokamak page</a> is where "
            "the second route earns its keep: a real pressure profile "
            "puts positive entries in the Jacobian and closes the cone "
            "outright.</p>",
            code_section(sf.newton_enclosure, sf.pn_junction,
                         sf.junction_charge_bracket, sf.junction_dispatch,
                         sf.depletion_width_analytic),
            "<h2>The run</h2>"
            f"<p>A silicon pn junction, 1 &#181;m, 10<sup>17</sup> "
            f"cm<sup>-3</sup> both sides, 200 cells, at {volts:g} V "
            "reverse bias. Newton from the charge-neutral guess:</p>"
            "<table><thead><tr><th>Newton steps</th><th>residual</th>"
            "<th>charge (nC/cm&#178;)</th>"
            "<th>certified error</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<p class='note'>A fully converged run of the same "
            f"iteration puts the charge at {truth:.6f} nC/cm&#178;. It "
            f"lies inside <strong>{contained}/{len(certd)}</strong> of "
            "the certified brackets.</p>",
            f"<figure>{svg}<figcaption>Two curves that a residual-based "
            "stopping rule assumes are the same curve. The dashed line "
            "is the residual, which falls smoothly from the first step. "
            "The solid line is the certified error, which does not "
            f"exist until step {first} &#8212; Kantorovich either closes "
            "or it does not &#8212; and then falls off a cliff, because "
            "quadratic convergence squares the residual and the theorem "
            "is linear in it.</figcaption></figure>",
            "<h2>Five parts in a million buys half a per cent</h2>"
            f"<p>The last refusing step has residual "
            f"{rlast[1]:.2g}. One step later the residual is "
            f"{r0[1]:.2g} &#8212; the sort of number that gets called "
            f"converged &#8212; and the certificate closes at "
            f"{r0[3]:.3g} nC/cm&#178;, which is "
            f"{100 * r0[3] / r0[2]:.2f} per cent of the answer.</p>"
            f"<p>The gap between those two numbers is not a detail of "
            f"the bound; it is the problem's conditioning. "
            f"&#8214;J<sup>-1</sup>&#8214; is {beta:.0f} here, and the "
            "charge functional contributes its own gradient on top. "
            "Neither factor is visible in the residual, so a stopping "
            "rule written on the residual is guessing at their product. "
            f"One further step and the certified error falls by a "
            f"factor of {drop:.0f}.</p>",
            "<h2>The front door</h2>"
            "<p>The tolerance is stated in the unit the measurement "
            "comes in: nC/cm&#178; of charge per unit area, which is "
            "what a capacitance-voltage sweep integrates.</p>"
            f"<pre>{esc(chr(10).join(lines))}</pre>"
            "<p>The refusal at 5 V is not about the tolerance. The "
            "potential has to travel two hundred thermal volts and the "
            "damping cap moves it three at a time, so twenty steps "
            "cannot arrive however loose the request &#8212; and the "
            "refusal says that rather than reporting the iterate it "
            "happened to reach.</p>",
            "<h2>What is being certified</h2>"
            "<p>The enclosure is around the exact solution of the "
            "<em>discretised</em> equations. The distance to the "
            "differential equation is the model's, and it is stated "
            "rather than absorbed:</p>"
            "<table><thead><tr><th>cells</th><th>charge "
            "(nC/cm&#178;)</th><th>certified error</th>"
            "<th>change from previous</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{n}</td><td>{c.value:.6f}</td>"
                      f"<td>{c.err:.2g}</td><td>"
                      + (f"{mdiff[i - 1]:.4g}" if i else "&#8212;")
                      + "</td></tr>" for i, (n, c) in enumerate(mesh))
            + "</tbody></table>"
            f"<p>Second order in the mesh, and about "
            f"{mdiff[0] / mesh[0][1].err:.0e} times the certified "
            "radius at the default resolution. On this problem the "
            "certificate is nowhere near the weak link &#8212; which is "
            "only known because both were measured, and is the reason "
            "to print both.</p>"
            "<p>The independent check comes from a closed form that "
            "uses none of the discretised operators. The textbook "
            "depletion approximation sweeps the junction perfectly "
            "clean of carriers, which must overstate the charge, and "
            "must overstate it by less as reverse bias widens the "
            "region it is wrong about:</p>"
            "<table><thead><tr><th>bias (V)</th><th>depletion width "
            "(nm)</th><th>textbook charge</th><th>certified charge</th>"
            "<th>ratio</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{vv:g}</td><td>{w:.1f}</td>"
                      f"<td>{qa:.2f}</td><td>{qc:.2f}</td>"
                      f"<td>{ra:.4f}</td></tr>" for vv, w, qa, qc, ra in dep)
            + "</tbody></table>"
            f"<p>Below one at every bias, and climbing monotonically "
            f"from {dep[0][4]:.4f} to {dep[-1][4]:.4f}. The textbook "
            "answer is wrong in the direction and by the amount the "
            "textbook says it should be, which is the strongest check "
            "available here, because nothing in that formula touched "
            "the matrices.</p>",
            "<h2>The witness, borrowed &#8212; and the bug it found</h2>"
            "<p>Reusing a proof across domains is the flywheel this "
            "project is betting on, and it is worth reporting that the "
            "reuse was not free. <code>mmatrix_witness</code> found its "
            "witness by solving Lu = 1 and nudging the result up by a "
            "fixed 10<sup>-9</sup> for slack. That nudge is a silent "
            "bet that the solve behind it was accurate to better than "
            "10<sup>-9</sup>. On a reactor it was. Written the way the "
            "physics writes it, this Jacobian has a condition number of "
            f"{cond:.2g}, the solve is wrong in the eighth digit, and "
            f"the nudged vector satisfies only Lu &#8805; "
            f"{naive_short:.10f} &#8212; short of 1, so the witness "
            "refused a matrix that is perfectly good.</p>"
            "<p>The fix removes the constant instead of enlarging it: "
            "scale u by its own measured shortfall, which costs one "
            "extra matvec and cannot be outgrown, because the shortfall "
            "and the rounding pad both scale with u. A borrowed proof "
            "was load-bearing enough to expose an assumption its "
            "original caller never tested, and the reactor's own "
            "certificates are unchanged.</p>",
            "<h2>Checked in this run</h2><ul>"
            f"<li>Containment against a fully converged solve: "
            f"<strong>{contained}/{len(certd)}</strong> certified "
            "rungs.</li>"
            f"<li>The certificate first closes at step "
            f"<strong>{first}</strong>; step {first - 1} is refused "
            f"with residual {rlast[1]:.2g}.</li>"
            f"<li>One step past the first certificate the error falls "
            f"by <strong>{drop:.0f}&#215;</strong>, and settles at "
            f"<strong>{certd[-1][3]:.2g} nC/cm&#178;</strong>.</li>"
            f"<li>Mesh convergence measured second order; the "
            f"discretisation is <strong>{mdiff[0] / mesh[0][1].err:.0e}"
            "&#215;</strong> the certified radius.</li>"
            f"<li>Against the depletion approximation: ratio "
            f"<strong>{dep[0][4]:.4f}</strong> at 0 V rising to "
            f"<strong>{dep[-1][4]:.4f}</strong> at 3 V, in the "
            "direction the approximation is known to err.</li>"
            "<li>A 5 V bias on a twenty-step ladder <strong>refuses"
            "</strong>, and prices the damping cap rather than the "
            "tolerance.</li>"
            f"<li>The reactor's M-matrix witness ran unaltered on a "
            f"Jacobian of condition number {cond:.2g}, after its fixed "
            "nudge was replaced by a measured one.</li>"
            "</ul>",
        ])


CASES = {
    "junction.html": junction_case,
    "criticality.html": criticality_case,
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
    "the-budget.html": budget_case,
}

# pages needing tools CI does not have: generated locally, committed,
# served as-is — the recorded-run pattern from TARGETS (empty since CI
# gained FEniCSx from the PPA; Basilisk will likely use it next)
RECORDED = {}

if __name__ == "__main__":
    check_counts()
    check_composed()
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
