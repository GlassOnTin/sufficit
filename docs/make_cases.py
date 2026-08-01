"""Basilisk-style case pages: each page is BUILT by running its case.
The source renders as the presentation (extracted live via inspect), the
a priori theory is stated, and the certification is executed and drawn.
Run from the repo root:  python3 docs/make_cases.py
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sufficit as sf  # noqa: E402


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tfi_case():
    # ---- run the case ----------------------------------------------------
    N = 10
    H0, H1 = sf.tfi_chain(N)
    thetas = np.linspace(0.0, 2.0, 6)
    sur = sf.reduced_basis_surrogate(H0, H1, thetas)
    gs = np.linspace(0.0, 2.0, 100)
    ups, lows, exact = [], [], []
    for g in gs:
        c = sf.reduced_basis_bracket(sur, g)
        ups.append(c.value + c.err)
        lows.append(c.value - c.err)
        exact.append(float(np.linalg.eigvalsh(H0 + g * H1)[0]))
    ups, lows, exact = map(np.array, (ups, lows, exact))
    contained = int(np.sum((lows <= exact) & (exact <= ups)))
    gap_at_crit = float(ups[np.argmin(np.abs(gs - 1.0))]
                        - exact[np.argmin(np.abs(gs - 1.0))])

    # ---- draw the certification ------------------------------------------
    W, Hpx, mL, mR, mT, mB = 640, 340, 58, 16, 18, 44
    ymin, ymax = float(lows.min()) - 0.5, float(ups.max()) + 0.5

    def X(g):
        return mL + (g / 2.0) * (W - mL - mR)

    def Y(e):
        return mT + (ymax - e) / (ymax - ymin) * (Hpx - mT - mB)

    def path(xs, ys):
        return "M " + " L ".join(f"{X(x):.1f} {Y(y):.1f}"
                                 for x, y in zip(xs, ys))

    band = ("M " + " L ".join(f"{X(g):.1f} {Y(u):.1f}"
                              for g, u in zip(gs, ups))
            + " L " + " L ".join(f"{X(g):.1f} {Y(l):.1f}"
                                 for g, l in zip(gs[::-1], lows[::-1]))
            + " Z")
    snaps = "".join(
        f'<circle cx="{X(t):.1f}" cy="{Y(l):.1f}" r="4.5" class="blue-fill"/>'
        for t, l in zip(sur["thetas"], sur["lows"]))
    yticks = "".join(
        f'<line x1="{mL}" y1="{Y(v):.1f}" x2="{W-mR}" y2="{Y(v):.1f}" '
        f'class="board-ink" opacity="0.25" stroke-width="1"/>'
        f'<text x="{mL-8}" y="{Y(v)+3.5:.1f}" text-anchor="end" '
        f'class="board-text" font-size="10.5" opacity="0.7">{v:g}</text>'
        for v in range(int(np.ceil(ymin)), int(ymax) + 1, 3))
    xticks = "".join(
        f'<text x="{X(v):.1f}" y="{Hpx-18}" text-anchor="middle" '
        f'class="board-text" font-size="10.5" opacity="0.7">g = {v:g}</text>'
        for v in (0.0, 0.5, 1.0, 1.5, 2.0))
    svg = f'''<svg viewBox="0 0 {W} {Hpx}" role="img"
  aria-label="Certified band around the exact ground energy of the
  transverse-field Ising chain across its quantum phase transition">
  {yticks}{xticks}
  <line x1="{X(1.0):.1f}" y1="{mT}" x2="{X(1.0):.1f}" y2="{Hpx-mB}"
        class="board-ink" stroke-dasharray="4 4" stroke-width="1.2" opacity="0.6"/>
  <text x="{X(1.0):.1f}" y="{mT-4}" text-anchor="middle" class="board-text"
        font-size="10.5" opacity="0.8">quantum phase transition</text>
  <path d="{band}" fill="var(--blue)" opacity="0.16" stroke="none"/>
  <path d="{path(gs, ups)}" fill="none" class="rust-ink" stroke-width="2"/>
  <path d="{path(gs, lows)}" fill="none" class="blue-ink" stroke-width="2"/>
  <path d="{path(gs, exact)}" fill="none" class="board-ink"
        stroke-width="1.6" stroke-dasharray="5 4"/>
  {snaps}
  <text x="{mL+6}" y="{mT+14}" class="board-text" font-size="11">
    ground energy, {N}-spin chain (hartree-like units)</text>
</svg>'''

    # ---- the code IS the presentation ------------------------------------
    src = "\n\n".join(inspect.getsource(f) for f in
                      (sf.tfi_chain, sf.reduced_basis_surrogate,
                       sf.reduced_basis_bracket))

    return f'''<title>Case: certified reduced basis across a phase transition</title>
{STYLE}
<main>
<p class="eyebrow"><a href="../index.html">sufficit</a> · certified case</p>
<h1>A certified parameter sweep across a quantum phase transition</h1>
<p class="dek">Six exact solves buy a rigorous two-sided band over a
100-point sweep — including the critical point, where nothing about the
physics is smooth.</p>

<h2>The theory, a priori</h2>
<p>For an affine family <code>H(g) = H₀ + g·H₁</code>, the ground energy
is an infimum of affine functions of <code>g</code>:
<code>λ(g) = min<sub>v</sub> [⟨v|H₀|v⟩ + g·⟨v|H₁|v⟩]</code> — and an
infimum of lines is <em>concave</em>. Concave functions lie above their
chords, so certified lower bounds at snapshot points certify the chord
between them, at every intermediate <code>g</code>, with no further solve.
The upper bound is the reduced-basis Rayleigh quotient — variational,
hence rigorous, and computable exactly at k×k cost from precomputed
Grams. Outside the snapshot hull the chord argument certifies nothing,
and the code refuses.</p>

<h2>The code (rendered from the module, not copied)</h2>
<pre>{esc(src)}</pre>

<h2>The certification, executed and drawn</h2>
<figure>{svg}
<figcaption>Blue dots: the 6 certified snapshot lower bounds (offline,
full solves). Blue curve: the chord lower bound. Rust curve: the
reduced Rayleigh upper bound. Dashed: the exact answer from full
1024-dimensional diagonalization at all 100 sweep points — computed only
to check the certificate, never used by it.</figcaption>
</figure>

<h2>Verified</h2>
<ul>
<li><strong>{contained}/100</strong> sweep points: exact energy inside the
certified band (this page is regenerated by running the case — a failed
containment would print here).</li>
<li>Upper-bound error at the critical point <code>g = 1</code>:
<strong>{gap_at_crit:.2e}</strong> — eigenvector continuation's engine:
six snapshot vectors span the ground state astonishingly well even where
correlations diverge.</li>
<li>Online cost per sweep point: one <strong>6×6</strong> eigenvalue
problem. The 1024-dimensional operator is never touched after the
offline stage.</li>
</ul>
<p style="color: var(--muted); font-size: 0.9rem;">This case is also
enforced in <code>test_sufficit.py</code> (containment at all 100 points,
refusal outside the hull, tightening with snapshot count).</p>
</main>'''


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
    overflow-x:auto; font-size:0.78rem; line-height:1.5;
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
  .rust-ink { stroke:var(--rust); }
  li { margin-bottom:0.5rem; }
</style>'''


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "cases"), exist_ok=True)
    out = os.path.join(here, "cases", "tfi-reduced-basis.html")
    with open(out, "w") as f:
        f.write(tfi_case())
    print(f"wrote {out}")
