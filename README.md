# sufficit

A query compiler for physics. Declare the model, the question, and the
error you can tolerate; get back a cheap computation and a proof of how
wrong it could be. The rewrites and their certificates are real, built
and proved by hand, one per problem — and the search half is growing: a
planner that selects and escalates among competing rewrites by declared
cost with the certificate as referee, a composed plan that splits one
error budget between two stages priced by a sensitivity the certificate
exports, and pipelines of different rewrites chained by exported
sensitivities — two stages for the equilibrium solve, three for the
smeared spectral value, where each resolution rung reprices the model
stage before it, and a fan-in where one budget divides between two
branches that never feed each other and no formula predicts what
either will certify. Those four specified a combinator, which now
exists: a plan is declared as stages and one assemble function, each
stage naming the stages it consumes, so the object is a graph; stages
that can solve for a knob given a budget are solved and the rest are
enumerated and walked cheapest-first. **All four are now wired through
it**, to the same knobs and the same costs the hand-written versions
chose, and the split and the binding stage are derived rather than
written down per front door. Hand-rolling one again is not a matter of
discipline: the planner refuses a multi-dimensional knob it did not
build, and the build refuses a front door that switches that check
off.

Wiring the last two is what finished the combinator, because they
needed what it lacked. A stage may now be a *choice* rather than a
certificate — picking an N-point kernel spends error budget on what
that kernel cannot resolve, and the bill lands inside somebody else's
certificate rather than in one of its own; two stages may share that
choice, which is how a model and a measurement live on one grid.

The other half of that debt is now closed by measurement rather than
by code. Three searches that use the graph's shape instead of only its
cost have been built and abandoned: escalating the binding stage,
ordering by marginal node cost, and pruning assignments whose measured
components already exceed the tolerance. The third is admissible and
cuts a five-branch walk from 1024 assemblies to 5, and it still saves
no nodes in nine of twelve cases. All three fail for the same reason,
which is worth knowing before a fourth is attempted: **the shared-node
memo already makes node computations linear in the graph**, so the only
quantity a cleverer search can reduce is assemblies, and assemblies are
arithmetic on certificates already in hand. A test pins that property,
because every one of those measurements depends on it.

Working research code: one module of certified rewrites
([`sufficit.py`](sufficit.py)) and one test suite
([`test_sufficit.py`](test_sufficit.py), 188 checks) in which every claimed
bound is verified against brute force, exact solutions, or independent
constructions. Every answer carries a value, an error bound, a tier, and
the provenance of the bound. The bound is the product.

New here? Start with the
**[tour](https://glassontin.github.io/sufficit/)**, then
[VISION.md](VISION.md), the founding argument, and [TARGETS.md](TARGETS.md),
the nine target domains. All nine now have entry rewrites, and two more have
been entered from that document's further-afield list — reactor criticality,
whose certificate comes from positivity rather than from a minimum principle,
and semiconductor devices, whose certificate proves that the answer exists
before saying how accurate it is. Each has a
[case page](https://glassontin.github.io/sufficit/)
that regenerates from a fresh run on every push.

[LEARNING.md](LEARNING.md) asks the question the other way round: what an
agent would be if it were built this way. It argues that the portable part
is the untrusted proposer and the cheap checker rather than the
certificates, that learning splits into three kinds with very different
risk, and that this project is an existence proof for only one of them.

## Two answers with receipts

```python
import sufficit as sf

# H2 molecule, from raw Gaussian integrals to a two-sided energy bracket
c = sf.h2_energy_bracket(1.4)          # bond length in bohr
print(c.value, c.err)                  # -1.137275944 +/- 1.5e-13 hartree

# A Heisenberg chain with 4^2000 quantum states, bracketed in ~1 second,
# because window cost does not depend on chain length
h = sf.heisenberg_chain_bracket(2000)  # per-bond: [-0.4572, -0.4221]
```

There is no package yet: the library is the single file
[`sufficit.py`](sufficit.py) — clone and import. Requirements: `numpy` and
`scipy`; `mpmath` for the tests; `cvxpy-base` and `scs` only for the SOS
search, whose result is re-proven exactly. Run the suite with
`pytest test_sufficit.py` (about four minutes; the SPH ladders are the
long pole).

## Measured results

| Problem | Result | Guarantee |
|---|---|---|
| N-body sums, 50k bodies | 64× fewer operations | pointwise within the requested ε |
| Black-box kernels (H-matrix + butterfly) | amortized applies, per-block competition | holds for every future input; failure odds 10⁻¹⁰, stated |
| 2D Ising at high temperature | free energy and correlations, floating point carried in intervals | refuses outside the proven convergence region |
| Helmholtz scattering | solver depth chosen from the requested ε | per-angle certified; refuses strong scattering |
| H₂ from scratch (McMurchie–Davidson, s+p) | −1.137275944 ± 1.5·10⁻¹³ Ha | two-sided bracket over all particle sectors |
| Hydrogen chains up to 2²⁰ states | 55 mHa/atom bracket at ℓ=7 | rigorous on both sides; no exact answer exists, none needed |
| Smeared spectral functions (HLT) | resolution is part of the query | error bounded by the data; degrades to statistics when the data are noisy |
| Mori–Zwanzig closures | certified linear tier and conformal empirical tier | gap-dependent bound, or distribution-free fail_p = 1/(n+1) |
| TFI quench on a 10⁶-site chain | ⟨Z(t)⟩ ± 10⁻³ in ~2 s | boundary commutator measured inside the cone; refuses when the light cone outruns the budget |
| Guiding-center drift (plasma hierarchy) | the first ASYMPTOTIC-tier certificate | exponent proven, constant measured on a cheap large-ε ladder; refuses when the data contradict the exponent |
| Lorenz ⟨z⟩ (SOS transport bound) | [27, 27.001], sharp to the fixed-point witness | Gram identity and positive-definiteness proven in exact rational arithmetic |
| Breaking wave on a sea wall (SPH) | delivered impulse certified; raw peak refused (no asymptotic range); a 40% berm certifiably zeroes the load | grid-convergence certificate with refusal and a capped measured order |
| Tokamak equilibrium (Grad–Shafranov via FEniCSx) | guaranteed energy-norm bound within 1.6× of the true error; implicit coupling certified by a contraction factor | Prager–Synge with rectangle-exact constants — guaranteed, not estimated; refuses past the contraction limit |
| Gravitational-wave surrogates | any parameter in ~0.3 ms | conformal mismatch bound with fail_p = 1/(n_cal+1); refuses outside the training range or above the detector's ε |
| Reactor criticality (k_eff) | 1 pcm in 20 fission-source iterations, bracket floor ~10⁻⁶ pcm; a mesh ladder then certifies the continuum answer to 0.57 pcm, checked against the closed-form buckling value | Perron–Frobenius, not the variational theorem: the operator is not self-adjoint, so positivity brackets it instead. Four hypotheses machine-checked; the solver's own error priced by the same witness. Continuum half degrades to EMPIRICAL and says so |
| The same bracket on S_N transport | runs unaltered on a different equation — no new proof, no new code | the certificate needs the operator to preserve a cone, not to be symmetric; upwind differencing supplies that. Having both models measures the diffusion approximation itself: −9200 pcm at 5 mean free paths, −41 at 20 |
| pn junction (drift-diffusion Poisson) | a proof that an exact solution exists within a stated radius, not just that the residual is small | Newton–Kantorovich, with ‖J⁻¹‖ priced by the reactor's M-matrix witness on completely different physics. The last refusing rung has residual 10⁻³; one step later, at 5·10⁻⁶, the certificate closes — and still only to half a per cent, which no residual reveals |
| The planner (first compiler slice) | one question at three tolerances → three algorithms; model-guided jump reaches the certifying rung in 4 runs where stepping takes 7 | cost models order the attempts, certificates arbitrate; every rung logs predicted cost, measured cost, and measured error — the receipt audits the cost model |

## How it stays honest

- **The check does not care how the guess was found.** Optimizers,
  sketches, and SDP solvers propose. Certificates come from the
  variational theorem, Cholesky feasibility, probe bounds with stated
  odds, or exact rational arithmetic. A bad guess costs tightness, never
  truth.
- **Three tiers, declared.** RIGOROUS: the bound is proven. ASYMPTOTIC:
  the exponent is proven and the constant is measured. EMPIRICAL: the
  guarantee is statistical and the failure probability is printed.
  Composition keeps the weakest tier.
- **Refusal is an answer.** Outside a proven validity region the
  functions raise, and the error message prices the next option.
- **Failures are kept as tests**, so a lesson is never bought twice: a
  demodulation that provably changed nothing, a rebalancing rule that
  measured 5 mHa worse, a warm start that loosened certificates from 66
  to 87 mHa.
- **Certificates bound the model's answer, not the world's.** The model
  is declared in the query. The H₂ bracket is ±10⁻¹³ hartree about the
  STO-3G model, which itself sits ~0.05 hartree from the molecule.
  Choosing a better model is a separate, stated step.

## Neighbors

None of the proof techniques are invented here. Ball arithmetic and
validated numerics (Arb, INTLAB) supply the interval habits; the
certifying-algorithms literature supplies the proposer/checker split; the
SOS bounds on time averages follow Tobasco, Fantuzzi, and Goulart; the
guaranteed FEM bounds follow Prager and Synge via the equilibrated-flux
literature; the distribution-free tier is conformal prediction. The bet
specific to Sufficit is composition: these traditions behind one query
interface, with one certificate type, so bounds from different proofs add
like autodiff gradients.

## Contributing

From [VISION.md](VISION.md): the most useful first contribution is an
attack. Find a rewrite whose claimed bound fails to compose, and file it
as an issue with a counterexample.

## License

[AGPL-3.0](LICENSE). Strong copyleft, network use included: improvements
to the rewrite library stay in the commons.
