# sufficit

*A query compiler for physics: declare the model, the question, and the error
you can tolerate; the system searches for the compression, the algorithm, and
the certificate.*

Working research code. One module of certified rewrites
([`sufficit.py`](sufficit.py)), one test suite in which every claimed bound is
checked against brute force, exact solutions, or independent constructions
([`test_sufficit.py`](test_sufficit.py) — 79 checks, several of which exist to
commemorate a mistake). **The certificate is the product**: every answer ships
as `value ± err` where the `±` is a theorem (or carries a printed failure
probability), never a hope.

New here? Start with the
**[illustrated tour](https://glassontin.github.io/sufficit/)**, then
[VISION.md](VISION.md) (the founding argument) and
[TARGETS.md](TARGETS.md) (where it should earn its keep).

## Two answers with receipts

```python
import sufficit as sf

# H2 molecule, from raw Gaussian integrals to a two-sided energy bracket
c = sf.h2_energy_bracket(1.4)          # bond length in bohr
print(c.value, c.err)                  # -1.137275944 +/- 1.5e-13 hartree

# A Heisenberg chain with 4^2000 quantum states — bracketed in ~4 seconds,
# because window cost is independent of chain length
h = sf.heisenberg_chain_bracket(2000)  # per-bond: [-0.4566, -0.4221]
```

Requirements: `numpy`, `scipy`, `mpmath` (tests only). Run the suite with
`pytest test_sufficit.py` (~1 minute).

## Measured highlights

| Problem | Result | Guarantee |
|---|---|---|
| N-body sums, 50k bodies (FMM rediscovered) | 64× fewer ops | pointwise ≤ requested ε |
| Black-box kernels (certified H-matrix + butterfly) | amortized applies, per-block rewrite competition | holds for every future input; `fail_p = 1e-10`, printed |
| 2D Ising at high temperature | free energy + correlations, FP error carried by intervals | refuses outside its proven validity region |
| Helmholtz scattering (certified far field) | Neumann depth chosen from your ε | per-angle certified; refuses strong scattering |
| H₂ from scratch (McMurchie–Davidson, s+p) | −1.137275944 ± 1.5e-13 Ha; 5.000 mHa polarization gain proven strictly | two-sided bracket over all particle sectors |
| Hydrogen chains, up to 2²⁰ states (H₁₀, ℓ=7) | 55 mHa/atom bracket | rigorous both sides — no exact answer exists, none needed |
| Smeared spectral functions (HLT mold) | resolution is part of the query | error bounded by the data itself; degrades honestly to statistics |
| Mori–Zwanzig closures | certified linear tier + conformal empirical tier | gap-dependent bound, or distribution-free `fail_p = 1/(n+1)` |

## How it stays honest

- **Guess freely, check strictly.** Optimizers, sketches, and bundles are never
  trusted; certificates are issued a posteriori by checks that cannot be argued
  with (variational theorem, Cholesky feasibility, probe bounds with stated
  odds). A bad guess costs tightness, never truth.
- **Tiers are declared.** `RIGOROUS` / `ASYMPTOTIC` / `EMPIRICAL` travel in the
  type and degrade to the weakest input. An empirical certificate honestly
  declared beats a rigorous one falsely implied.
- **Refusal is an answer.** Outside a proven validity region (cluster-expansion
  radius, weak-scattering norm, spectral gap), functions raise instead of
  extrapolating.
- **Negative results are kept as tests**, so they are never expensively
  re-learned: directional demodulation is unitarily vacuous (σ-values
  identical, proven and pinned); the textbook ε-rebalancing rule is not an
  ascent step (measured −5 mHa/atom); Lanczos warm starts bias cuts (measured
  66→87 mHa/atom).

## Contributing

From [VISION.md](VISION.md): the first useful contribution is an attack — a
rewrite whose claimed bound fails to compose, filed as an issue with a
counterexample.
