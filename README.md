# sufficit

A query compiler for physics. Declare the model, the question, and the
error you can tolerate. Get back a cheap computation and a proof of how
far it can sit from the model's exact answer.

Every error must live in one of two places. Error from computing (a
mesh, a truncation window, a finite sample, a float) goes in the
certificate: bounded, tiered, with provenance. Error from modelling (a
2D tank standing in for a 3D sea, a box of 216 molecules, a basis set)
goes in the declaration: named, never bounded. Nothing floats between,
so after a run the only thing left to doubt is the model, and the model
is written down.

Working research code: one module of certified rewrites
([`sufficit.py`](sufficit.py)) and one test suite
([`test_sufficit.py`](test_sufficit.py), 247 checks). Every claimed bound is
verified there against brute force, exact solutions, or independent
constructions.

New here? Start with the
**[tour](https://glassontin.github.io/sufficit/)**. Then read
[VISION.md](VISION.md) for the founding argument and
[TARGETS.md](TARGETS.md) for the nine target domains. All nine have entry
rewrites. Two more have been entered from that document's further-afield
list: reactor criticality, whose certificate comes from positivity rather
than from a minimum principle, and semiconductor devices, whose certificate
proves that the answer exists before saying how accurate it is. Each domain
has a [case page](https://glassontin.github.io/sufficit/) that regenerates
from a fresh run on every push.

[LEARNING.md](LEARNING.md) asks the question the other way round: what an
agent would be if it were built this way. It argues three things. The
portable part is the untrusted proposer and the cheap checker, not the
certificates. Learning splits into three kinds with very different risk.
This project is an existence proof for only one of them.

## Ask a question, get a receipt

```python
import sufficit as sf

# Declare the question and the tolerance. The planner picks the algorithm.
c = sf.h_chain_energy_dispatch(6, tol=0.08)   # H6 ground energy, STO-3G,
c.value, c.err, c.tier.name                   #   tol is half-width/atom
# (-3.3524, 0.2659, 'RIGOROUS')
c.provenance[0]
# 'h-chain marginal-lower ell=4 n=6 d=1.8 iters=60 block-product-upper ...'

# Too tight a tolerance is refused, and the refusal is a receipt: every
# rung it ran, predicted against measured cost, and the price of the
# cheapest thing it did not try.
sf.h_chain_energy_dispatch(6, tol=1e-6)
# Refusal: hchain-energy: no rung within budget certifies tol=6e-06 at
# n=6; measured (knob, err, cost): (2, 2.32, 0.21s), (3, 0.715, 0.26s),
# (D, 1.24, 0.13s), (5, 0.105, 1.2s), (DQ, 0.093, 2.1s), (DQG, 0.0499,
# 10s); the next window ell=6 costs 4^6 = 4096 and is past the declared
# ladder (ell_max=5)

# Scale is a knob, not a wall: a Heisenberg chain with 4^2000 quantum
# states, bracketed in three seconds, because window cost does not
# depend on chain length. Per bond: [-0.4566, -0.4221], RIGOROUS.
h = sf.heisenberg_chain_bracket(2000)
```

There is no package yet. The library is the single file
[`sufficit.py`](sufficit.py), so clone and import. Requirements: `numpy` and
`scipy`; `mpmath` for the tests; `cvxpy-base` and `scs` only for the SOS
search and the 2-RDM bound, both of which re-certify the solver's
output on our side. Run the suite with
`pytest test_sufficit.py`. It takes three and a half minutes on an idle
32-core box, about six on an older machine, and noticeably longer on a
busy one. The SPH ladders and the planner's receipt test are the long
poles.

## What comes back

One certificate type serves every domain. A `Certified` carries:

- `value`: the answer, float or array.
- `err`: a bound on the distance from `value` to the declared model's
  exact answer.
- `tier`: `RIGOROUS` (the bound is proven), `ASYMPTOTIC` (the exponent
  is proven, the constant is measured), or `EMPIRICAL` (statistical,
  with the failure probability carried in `fail_p`).
- `provenance`: which rewrites produced the bound, with their knobs.
- `sensitivity`: when the rewrite can certify one, a Lipschitz bound on
  how hard the output leans on a named input, so a composed plan can
  price the input error it feeds in. `None` means no claim, never no
  amplification.

Bounds compose the way gradients do in autodiff: errors add, tiers take
the minimum, failure probabilities union-bound. When no rung within
budget certifies the tolerance, the front door raises a `Refusal` that
records every rung it ran, predicted cost beside measured cost, and the
price of the cheapest thing it did not try. The bound is the product.

## Will it take your problem?

The eleven domains below are demonstrations. The capability is seven
certificate archetypes, and the question is which structure your
problem exposes:

1. **Small-parameter expansions.** A proven exponent in a stated limit;
   the constant is measured on a cheap ladder (guiding-center drift,
   EFT-like hierarchies). ASYMPTOTIC tier.
2. **Variational sandwiches.** A minimum principle gives one side and a
   trial state the other (energy brackets, SOS transport bounds).
   RIGOROUS.
3. **Resolution-limited queries.** The instrument's finite resolution
   is part of the question, so the bound follows the data (smeared
   spectral functions, GW mismatch).
4. **Projection with memory.** Slow variables plus a closure that is
   certified where linear and honestly statistical where not
   (Mori-Zwanzig). EMPIRICAL, `fail_p` printed.
5. **Cone-preserving brackets.** The operator preserves a cone, so
   positivity brackets an eigenvalue the variational theorem cannot
   reach (reactor criticality, S_N transport). RIGOROUS.
6. **Existence certificates.** Newton-Kantorovich proves a solution of
   the nonlinear system exists near the iterate, before anything is
   said about accuracy (pn junctions, peaked tokamak profiles).
   RIGOROUS.
7. **Spectral enclosures.** A region containing every eigenvalue, from
   the coefficients alone, for operators with neither symmetry nor a
   cone. RIGOROUS, and loose.

If your problem exposes none of these, there is no certificate here for
it yet. [TARGETS.md](TARGETS.md) records the entries and the measured
dead ends alike; combustion's ignition delay is refused there with the
amplification factor that refuses it.

## Measured results

Each linked row has a case page that regenerates from a fresh run on
every push; the narrative and the failure modes live there. Results too
heavy for a page build (a 36 GB window, hour-long SPH rungs, hours of
molecular dynamics) are recorded by `docs/record.py` into `data/` with
provenance, restored and re-certified at build time, and named in the
page's footer.

| Problem | Result | Guarantee |
|---|---|---|
| N-body sums, 50k bodies | 64× fewer operations | pointwise within the requested ε |
| [Black-box kernels (H-matrix + butterfly)](https://glassontin.github.io/sufficit/cases/butterfly-crossover.html) | amortized applies, per-block competition | holds for every future input; failure odds 10⁻¹⁰, stated |
| [2D Ising at high temperature](https://glassontin.github.io/sufficit/cases/ising-cluster.html) | free energy and correlations, floating point carried in intervals | refuses outside the proven convergence region |
| Helmholtz scattering | solver depth chosen from the requested ε | per-angle certified; refuses strong scattering |
| [H₂ from scratch (McMurchie-Davidson, s+p)](https://glassontin.github.io/sufficit/cases/h2-bracket.html) | −1.137275944 ± 1.5·10⁻¹³ Ha | two-sided bracket over all particle sectors |
| [Hydrogen chains up to 2²⁰ states](https://glassontin.github.io/sufficit/cases/hchain-ladder.html) | 28 mHa/atom bracket at ℓ=8 | rigorous on both sides; no exact answer exists, and none is needed |
| [Smeared spectral functions (HLT)](https://glassontin.github.io/sufficit/cases/smeared-spectral.html) | resolution is part of the query | error bounded by the data; degrades to statistics when the data are noisy |
| [Mori-Zwanzig closures](https://glassontin.github.io/sufficit/cases/mz-closure.html) | certified linear tier and conformal empirical tier | gap-dependent bound, or distribution-free fail_p = 1/(n+1) |
| [TFI quench on a 10⁶-site chain](https://glassontin.github.io/sufficit/cases/lr-dispatch.html) | ⟨Z(t)⟩ ± 10⁻³ in ~2 s | the boundary commutator is measured inside the cone; refuses when the light cone outruns the budget |
| [Guiding-center drift (plasma hierarchy)](https://glassontin.github.io/sufficit/cases/plasma-hierarchy.html) | the first ASYMPTOTIC-tier certificate | exponent proven, constant measured on a cheap large-ε ladder; refuses when the data contradict the exponent |
| [Lorenz ⟨z⟩ (SOS transport bound)](https://glassontin.github.io/sufficit/cases/sos-transport.html) | [27, 27.001], sharp to the fixed-point witness | Gram identity and positive-definiteness proven in exact rational arithmetic |
| [Breaking wave on a sea wall (SPH)](https://glassontin.github.io/sufficit/cases/sph-wall.html) | delivered impulse certified, raw peak refused, and a 40% berm certifiably zeroes the load | grid-convergence certificate with refusal and a capped measured order |
| [Tokamak equilibrium (Grad-Shafranov via FEniCSx)](https://glassontin.github.io/sufficit/cases/gs-equilibrium.html) | guaranteed energy-norm bound within 1.6× of the true error | Prager-Synge with rectangle-exact constants; refuses past the contraction limit |
| [The same equilibrium with a real pressure profile](https://glassontin.github.io/sufficit/cases/gs-equilibrium.html) | certified where the contraction refuses, at a contraction factor of 33 against a limit of 1 | a peaked profile closes both the contraction and the cone route to ‖J⁻¹‖; Kantorovich certifies anyway, reaches the discrete equilibrium, and prints the mesh gap beside it |
| The continuum answer, in three domains | k_eff, junction charge and tokamak flux each lifted off their meshes by one shared step | the distance to h→0 is measured off the ladder, so the pair takes the weaker tier and is never RIGOROUS; checked against closed forms twice and out of sample once |
| [Gravitational-wave surrogates](https://glassontin.github.io/sufficit/cases/gw-surrogate.html) | any parameter in ~0.3 ms | conformal mismatch bound with fail_p = 1/(n_cal+1); refuses outside the training range or above the detector's ε |
| [Reactor criticality (k_eff)](https://glassontin.github.io/sufficit/cases/criticality.html) | 1 pcm in 20 fission-source iterations; a mesh ladder then certifies the continuum answer to 0.57 pcm | Perron-Frobenius positivity rather than the variational theorem, four hypotheses machine-checked; the continuum half degrades to EMPIRICAL and says so |
| [The same bracket on S_N transport](https://glassontin.github.io/sufficit/cases/criticality.html) | runs unaltered on a different equation, with no new proof and no new code | the certificate needs a preserved cone, not symmetry; having both models measures the diffusion approximation itself: −9200 pcm at 5 mean free paths, −41 at 20 |
| [pn junction (drift-diffusion Poisson)](https://glassontin.github.io/sufficit/cases/junction.html) | a proof that an exact solution exists within a stated radius, then accuracy to half a per cent | Newton-Kantorovich, with ‖J⁻¹‖ priced by the reactor's M-matrix witness on completely different physics |
| [The 2-RDM lower bound (chemistry)](https://glassontin.github.io/sufficit/cases/rdm2-bound.html) | H₁₄ certified in 497 s at 2²⁸ Fock states, where the matrix cannot be formed; DQG width 8.0 mHa/atom at H₄ barely moves to 8.9 at H₁₂ | the 2-positivity set contains every N-representable 2-RDM, so its minimum lies below the true one; SCS proposes the multipliers and nothing trusts them |
| [A molecular front door, on a real basis](https://glassontin.github.io/sufficit/cases/molecule.html) | σ-polarised H₆ is a 16,777,216-state Fock space, past forming, certified in 54 s; STO-3G water lands 2.3 mHa below its own full CI | two rewrites on one query, priced in orbitals; the basis constants are gated on three closed-shell molecules nobody here chose |
| [Water's density maximum (mW)](https://glassontin.github.io/sufficit/cases/water-tmd.html) | bracketed at 250 ± 20 K by a 216-molecule ladder, the published value at its centre; a separately annealed 512-molecule run reproduces it at 260 ± 30 K | certified means of correlated series and a certified argmax, family-wise 0.05; the cold rung was refused for run length, and spending the budget showed the refused number was nearly two half-widths wrong |
| [The planner (first compiler slice)](https://glassontin.github.io/sufficit/cases/the-compiler.html) | one question at three tolerances gets three algorithms; a model-guided jump reaches the certifying rung in 4 runs where stepping takes 7 | cost models order the attempts and certificates arbitrate; every rung logs predicted cost, measured cost, and measured error, so the receipt audits the cost model |

## The search half

Every rewrite above was built and proved by hand, one per problem. Choosing
between them is newer work.

A planner selects among competing rewrites by their declared cost and
escalates when the answer is not good enough. The certificate is the
referee. A plan can also be built from stages, so one error budget is
divided across several steps. There are four such plans. The equilibrium
solve uses two stages. The smeared spectral value uses three, where each
resolution rung reprices the model stage before it. The fourth divides one
budget between two branches that never feed each other, and no formula
predicts what either will certify.

Those four specified a combinator, which now exists. A plan is declared as
stages plus one assemble function, and each stage names the stages it
consumes, so a plan is a graph. Stages that can solve for a knob given a
budget are solved. The rest are enumerated and walked cheapest first.

All four plans now run through the combinator. They reach the same knobs and
the same costs the hand-written versions chose. The budget split and the
binding stage are derived rather than written down once per front door.
Hand-rolling one again is not a matter of discipline: the planner refuses a
multi-dimensional knob it did not build, and the build refuses a front door
that switches that check off.

Wiring the last two plans is what finished the combinator, because they
needed what it lacked. A stage may now be a choice rather than a
certificate. Picking an N-point kernel spends error budget on what that
kernel cannot resolve, and the bill lands inside another stage's certificate
rather than in one of its own. Two stages may share that choice, which is
how a model and a measurement come to live on one grid.

The rest of that debt is closed by measurement rather than by code. Three
searches that use the graph's shape instead of only its cost were built and
abandoned: escalating the binding stage, ordering by marginal node cost, and
pruning assignments whose measured components already exceed the tolerance.
The third is admissible and cuts a five-branch walk from 1024 assemblies to
5, and it still saves no nodes in nine of twelve cases. All three fail for
the same reason, which is worth knowing before a fourth is attempted. The
shared-node memo already makes node computations linear in the graph, so the
only quantity a cleverer search can reduce is assemblies, and assemblies are
arithmetic on certificates already in hand. A test pins that property,
because every one of those measurements depends on it.

## How it stays honest

- **The check does not care how the guess was found.** Optimizers,
  sketches, and SDP solvers propose. Certificates come from the
  variational theorem, Cholesky feasibility, probe bounds with stated
  odds, or exact rational arithmetic. A bad guess makes the bound looser.
  It cannot make the bound wrong.
- **Three tiers, declared.** RIGOROUS: the bound is proven. ASYMPTOTIC:
  the exponent is proven and the constant is measured. EMPIRICAL: the
  guarantee is statistical and the failure probability is printed.
  Composition keeps the weakest tier.
- **Refusing is a valid answer.** Outside a proven validity region the
  functions raise, and the error message prices the next option.
- **Failures are kept as tests**, so the same mistake is not made twice.
  A demodulation that provably changed nothing, a rebalancing rule that
  measured 5 mHa worse, and a warm start that loosened certificates from
  66 to 87 mHa are all still in the suite.
- **Certificates bound the model's answer, not the world's.** The model
  is declared in the query. The H₂ bracket is ±10⁻¹³ hartree about the
  STO-3G model, which itself sits about 0.05 hartree from the molecule.
  Choosing a better model is a separate, stated step.

## Neighbors

None of the proof techniques are invented here. Ball arithmetic and
validated numerics (Arb, INTLAB) supply the interval habits. The
certifying-algorithms literature supplies the proposer and checker split.
The SOS bounds on time averages follow Tobasco, Fantuzzi, and Goulart. The
guaranteed FEM bounds follow Prager and Synge by way of the
equilibrated-flux literature. The distribution-free tier is conformal
prediction. The bet specific to Sufficit is composition: these traditions
behind one query interface, with one certificate type, so that bounds from
different proofs add like autodiff gradients.

## Contributing

From [VISION.md](VISION.md): the most useful first contribution is an
attack. Find a rewrite whose claimed bound fails to compose, and file it
as an issue with a counterexample.

## License

[AGPL-3.0](LICENSE). Strong copyleft, network use included, so improvements
to the rewrite library stay in the commons.
