# Targets

Applied domains where the compiler should earn its keep. This is a companion to
`VISION.md`, whose vocabulary is assumed throughout: rewrite, certificate, and
the **rigorous / asymptotic / empirical** guarantee tiers.

## How a domain earns a slot

1. **The queries are low-dimensional functionals with tolerances the community
   already states.** Chemical accuracy, noise-weighted mismatch, FLAG-style
   error budgets. A target is only as good as its habit of declaring ε.
2. **A physical contraction with quantifiable provenance exists.** A gap, a
   small parameter, a light cone, a convergence radius, a resolution scale.
   Something a certificate can be hung on.
3. **At least one artisanal compression already works.** These become seed
   rewrites and regression tests. Per `VISION.md`, rediscovery is the test
   suite.
4. **A Python-facing ecosystem exists to compile against.** The compiler
   targets existing engines. It does not rewrite a field's codes.

Targets are ordered roughly by proximity to the Phase 0 to 3 machinery.

## Wave scattering and computational electromagnetics

The nearest beachhead, because it is Phase 1 wearing engineering clothes. The
queries are S-parameters, radar cross sections, antenna patterns, and photonic
band structures. Each is a handful of numbers per design, demanded across
thousands of parameter points in inverse-design loops.

The contraction is rank structure. There are hierarchical low-rank blocks for
Laplace-like regimes, directional and butterfly structure in the high-frequency
Helmholtz regime, and certified reduced bases across parameter sweeps where
inf-sup constants cooperate (Patera-Rozza a posteriori estimators). MLFMA and
butterfly-compressed boundary elements are the artisanal compressions to
rediscover.

The compiler's addition is automatic selection among H-matrix, butterfly and
reduced-basis rewrites per frequency band, with a composed bound feeding the
design loop. Certificate outlook: rigorous to asymptotic, which is the
friendliest terrain we have. Ecosystem hooks: FEniCSx, MEEP, scikit-rf.

## Plasma kinetics and fusion energy

Nobody wants f(x, v, t). Everyone wants its moments: turbulent heat and
particle fluxes, growth rates, transport coefficients, stability margins.

The field is organized around a hierarchy of small-parameter reductions,
running kinetic to gyrokinetic to gyrofluid to MHD, with explicit expansion
parameters such as the normalized gyroradius ρ* ~ 10⁻³ and collisionality. That
is to say asymptotic certificates already exist, hand-derived. Dynamical
low-rank approximation for Vlasov (Einkemmer-Lubich) brings genuine, if
partial, error analysis. The Hammett-Perkins Landau-fluid closure is a
hand-built Mori-Zwanzig projection and a canonical Phase 4 regression test.

The compiler's addition is certified dispatch along the reduction hierarchy per
query tolerance, plus closure discovery with an honestly declared empirical
tier. That is exactly what stellarator optimization loops need, since they can
only afford query-budgeted physics. Certificate outlook: asymptotic for the
hierarchy, empirical for discovered closures. Ecosystem hooks: pyrokinetics,
PlasmaPy, DESC.

## Quantum chemistry and materials

The flagship for rigorous intervals. Queries come with a declared tolerance
culture, namely chemical accuracy at 1 kcal/mol, and the contraction is
everywhere: Kohn nearsightedness in gapped systems, locality of correlation
(DLPNO-class methods reach near-linear scaling), and pervasive low-rank
structure in the two-electron integrals (density fitting, Cholesky, tensor
hypercontraction).

The headline product writes itself. A variational wavefunction gives an upper
bound. A variational 2-RDM under N-representability conditions gives an SDP
lower bound. The pair yields a *certified bracket on the ground-state energy
within a basis*. Basis-set incompleteness is then handled at the asymptotic
tier by extrapolation, or at the empirical tier by F12, and it is declared
rather than hidden.

The lower half of that pair now exists. The 2-positivity relaxation is
solved and certified in `rdm2_energy_bracket`, and spin blocking carries it to
H₁₄, which is 2²⁸ Fock states and well past the point where the Hamiltonian
can be formed. It certifies the N-electron sector rather than the whole Fock space,
which is a narrower claim than the bracket beside it. Its upper half is a
self-consistent determinant, so the sandwich is assembled from two proofs that
know nothing about each other, and the planner races it against the window
ladder. There is now a molecular front door too, on McMurchie-Davidson
integrals and published STO-3G for H, C, N and O. Polarisation functions push
a six-atom hydrogen cluster past the point where its Fock sector can be formed
and only the relaxation answers, and on water, where the exact answer is still
affordable, the relaxation lands 2.3 mHa below full CI.

The compiler's addition is replacing the folklore method ladder with
per-system, per-query selection of thresholds and rewrites against a composed
error budget. Certificate outlook: rigorous intervals where the sandwich
closes, and declared-empirical DFT elsewhere. Ecosystem hooks: PySCF, ASE.

## Lattice field theory

Home turf for Phases 2 and 3, extended to the queries the community actually
fights over. Window quantities for the muon g−2 hadronic vacuum polarization,
and HLT-smeared spectral functions, are resolution-limited queries by
construction. Multigrid Dirac solvers, multilevel Monte Carlo via domain
factorization (Cè-Giusti-Schaefer), and master-field techniques are the
existing hierarchy-exploiting rewrites.

The field's error-budget culture makes adoption plausible, since FLAG averages
institutionalize it. The compiler's addition is composing statistical sampling
error, solver tolerances, and continuum and volume extrapolations into one
end-to-end budget per declared query. Certificate outlook: statistical
certificates with rigorous components. The sign wall at finite density is a
mapped boundary, not a promise. Ecosystem hooks: gpt (Grid Python Toolkit).

## Nuclear structure and dense matter

Chiral EFT hands us the rare thing, an expansion parameter whose truncation
error is *modelable*. The BUQEYE program turns EFT convergence patterns into
statistical certificates with declared priors.

On the many-body side, eigenvector continuation revealed that parametric
nuclear solves live on tiny reduced bases, with a posteriori error estimation
emerging from its reformulation as a reduced-basis method. SRG evolution is a
softening rewrite that trades operator complexity for tractability.

The queries with leverage are binding energies, astrophysical reaction rates,
and neutron-star equation-of-state bands, where certified emulators must
propagate EFT truncation error all the way to observables. Certificate outlook:
statistical and asymptotic, unusually well-founded for a strongly correlated
field. Ecosystem hooks: the BUQEYE toolkits.

## Molecular kinetics, free energies, and coarse-graining

The queries are thermodynamic and kinetic functionals at kcal/mol tolerances:
binding free energies, solvation, rates, mechanisms.

Metastability is a spectral gap of the transfer operator, so Markov state
models are projections whose quality is bounded by variational scores (VAC and
VAMP). MBAR carries asymptotic variance estimates. Polymer and solvent
coarse-graining is Mori-Zwanzig in production use.

The compiler's addition is automated slow-variable identification with
variational certificates on the projection. It also adds, critically, an honest
empirical tier for force-field error, which dominates real budgets and is today
almost never propagated. This is Phase 4 territory with the best-developed
variational scaffolding. Certificate outlook: variational bounds on kinetics,
statistical on sampling, declared-empirical on the model itself. Ecosystem
hooks: OpenMM, deeptime, pymbar.

### The standing challenge: water's density maximum

The named goal for this column is to certify the temperature of maximum
density of a declared water model. Water is densest at about 3.98 °C at one
atmosphere, because the collapse of the open tetrahedral hydrogen-bonded
network competes with ordinary thermal expansion, and below that temperature
the collapse wins. It is the sharpest test of a force field anyone routinely
runs, and it is a good target here for three separate reasons.

It punishes the model specifically. Rigid three-site models miss the maximum by
tens of kelvin, and TIP3P has no maximum in the liquid range at all.
TIP4P/2005 and TIP5P land within a degree or two, on the same functional form
and at the same cost. Getting the density right at 25 °C tells you almost
nothing about whether a model will do this.

It is a sampling problem before it is a physics problem. Between 0 and 4 °C the
density moves by about one part in ten thousand, so the statistical error on an
NPT average has to sit well below that before the question can even be asked.

And the query is a stationary point, not a value, which changes what has to be
certified. Locating a maximum means certifying an ORDERING: rho(T1) < rho(T2) >
rho(T3), with the certified intervals disjoint enough that the ordering is
forced. The precision needed is set by the curvature of rho(T) near the peak
rather than by any tolerance stated up front, and precision at the peak trades
against width in T. That is a well-posed query in this library's terms and a
better-behaved one than asking for the density itself.

The error budget splits three ways and only two of them are ours. Statistical
error from finite sampling is certifiable, at the empirical tier, with a
printed failure probability. Finite-size error is a system-size ladder read by
the same grid-convergence machinery the reactor, the junction and the tokamak
already share, so it is an archetype transfer rather than new work. Force-field
error is neither, and it dominates both by orders of magnitude. So the
achievable claim is "the maximum-density temperature of TIP4P/2005 is 278 +- 2
K", and the unachievable one is "the maximum-density temperature of water is
277.13 K". The gap between those two sentences is exactly what the tier system
exists to declare rather than hide, and here it is tens of kelvin wide
depending on which model was declared.

The blocks, cheapest and most reusable first. The first two need no molecular
dynamics and no new dependency, and both can be gated against a truth known by
construction, so the certification machinery gets tested before an engine is
involved.

1. **A certified mean of a correlated time series. DONE**, as
   `timeseries_mean`. The bottleneck, and needed by every molecular-dynamics
   observable rather than just this one. An NPT average is over autocorrelated
   samples, so bounding it needs an effective sample size and a statement that
   survives correlation. Rigorously that wants a spectral gap nobody will have,
   so it lands at the empirical tier with a declared stationarity assumption.
   Measured over 300 AR(1) replicas of 20,000 samples, coverage runs 0.946
   against a nominal 0.95 at correlation times of 1, 9 and 39, and about 1% of
   series are refused.

   The refusal is the interesting half. The integrated autocorrelation time is
   estimated on a ladder of batch sizes, and if it has not plateaued the series
   is too short to bound the mean at all, which is the same refusal the sea
   wall makes when a resolution ladder shows no asymptotic range. Two
   measurements changed how that test is written. A fixed plateau tolerance
   fired on estimator noise and refused a quarter of INDEPENDENT series, so the
   threshold now carries the noise of the rungs being compared. And the plateau
   test alone was still too weak: at a correlation time of 199, where the
   longest affordable batch is about three correlation times, it let 64% of
   series through and those covered 0.901 rather than 0.95. It is only a proxy
   for what actually matters, so the direct requirement is now stated as well,
   that a batch be at least ten correlation times long. That refuses the whole
   under-covering regime, 300 out of 300. A drifting series, which is what an
   unequilibrated trajectory looks like, is refused for the same reason.
2. **A certified stationary point.** The ordering argument above, composed from
   three certified means. Gate: a synthetic rho(T) with a maximum put in by
   hand.
3. **A finite-size ladder.** `continuum_limit` already does this shape.
4. **An untrusted molecular-dynamics engine.** OpenMM proposes, we certify, on
   the same line every other engine here sits on.
5. **The query end to end**, against a model whose maximum-density temperature
   is published.

The model temperatures quoted above are recollection and are not gated. This
project has been bitten both ways in one sitting: the STO-3G exponents were
recalled correctly and passed their gate, and the N2 energy was recalled
correctly while the code around it was wrong. Nothing here should be repeated
as measured until it has been run.

## Quantum dynamics and near-term quantum hardware

The query with commercial teeth is dispatch. Given a circuit, an observable, a
time, and a noise model, does a classical path exist, and if so with what
certificate?

The compressible islands are mapped and growing. Lieb-Robinson cones bound
operator spreading rigorously. Noise truncates Pauli-path expansions, which is
the mechanism behind the classical reproduction of utility-scale experiments
and the Aharonov et al. dequantization. Stabilizer-rank decompositions
(Bravyi-Gosset) cover near-Clifford circuits. Process tensors are low-rank in
memory time (TEMPO). Classical shadows budget the measurement side.

The compiler's addition is turning this hardness cartography into a decision
procedure: a certified classical answer when one exists, and a quantum resource
estimate when not. Certificate outlook: rigorous on the islands, empirical for
heuristic tensor networks in between, with the tier printed on every answer.
Ecosystem hooks: QuTiP, stim, quimb.

## Turbulence and engineering transport

Eddies are detail. The queries are means: drag coefficients, Nusselt numbers,
mixing rates. There are two complementary seams.

First, the background-flow method in its modern sum-of-squares form delivers
*rigorous* bounds on mean quantities via convex optimization, following the
Doering-Constantin lineage through Tobasco, Fantuzzi and Goulart. Those
certificates are literally the product, on a problem famous for having none.

Second, RANS and subgrid closures are Mori-Zwanzig instances currently shipped
with silent extrapolation. Automated closure discovery with declared validation
domains is the honest upgrade.

The compiler's addition is pairing the two, so every reported mean carries
either a bound or a tier. Certificate outlook: rigorous bounds on one side,
declared-empirical closures on the other, and no pretense that they meet in the
middle yet. Ecosystem hooks: Dedalus.

## Gravitational-wave source modeling

The purest resolution-limited query in physics. Waveforms need only match to
the detector's noise-weighted mismatch, a tolerance set by instrument and SNR
rather than by formalism.

The existing compressions are reduced-order surrogates of numerical relativity,
built by empirical interpolation over a low-dimensional waveform manifold, and
post-Newtonian and EOB expansions, which sit at the asymptotic tier with
explicit small parameters. Accuracy studies against NR are the current, largely
empirical, certification practice.

The compiler's addition is mismatch-budgeted surrogate synthesis with
interpolation-error certificates where the manifold structure permits them.
That is query-first accounting as the literal design spec. Certificate outlook:
empirical with asymptotic components, and an unusually crisp ε. Ecosystem
hooks: bilby, gwsurrogate.

## Further afield

The same archetypes recur wherever detail is simulated but functionals are
wanted: combustion mechanism reduction (CSP and DRG against ignition-delay
queries), climate and weather subgrid closures, neutron transport and reactor
design, semiconductor device modeling. Each is admissible the moment it
satisfies the four criteria above. None is a launch target.

**Neutron transport** was entered first, ahead of the others and out of order,
because it pays a debt the nine could not. Every eigenvalue bracket in the
library ran on the variational theorem, which needs a self-adjoint operator. A
reactor down-scatters and is not one. The criticality eigenvalue is bracketed
by positivity instead, as the archetype below describes. The field states its
tolerance in pcm, so the certificate lands in units the culture already reads.

**Semiconductor devices** came next, and paid a different debt. Every
certificate here answered *how far is this number from the right one*, and none
could answer *is there a right one*. A device simulator runs Newton's method on
equations whose carrier densities are exponentials of the unknown, and stops
when the residual looks small. That is consistent with a solution nearby, with
a solution far away, and with none at all. Kantorovich's theorem converts a
residual into existence plus an enclosure. The constant it needs most is a
bound on the inverse Jacobian, which the reactor's M-matrix witness already
supplies, because the discretized nonlinear Poisson operator is a Z-matrix too.
Measured on a pn junction, the residual at the last refusing rung is 10⁻³. One
Newton step later it is 5·10⁻⁶ and the certificate closes, at half a per cent
of the answer, because ‖J⁻¹‖ is 237 and the residual shows none of that.

**Combustion mechanism reduction** was probed and **not** entered, which is
worth recording with its numbers rather than leaving as a gap. The headline
query is ignition delay, and a certificate on it must survive the
chain-branching phase. Measured on an eight-species hydrogen mechanism at
1100 K, the spectral abscissa integrates to ∫α dt ≈ 17 over the induction
period. Perturbations are genuinely amplified 2.5·10⁷ times, and that is what
ignition *is*, not a defect of any bound. A fixed-norm Gronwall bound pays far
more than the physical rate, giving ∫μ dt ≈ 3000 in the natural scaling and
≈ 67 after optimal diagonal balancing, because chemical Jacobians are extremely
non-normal. A componentwise comparison enclosure, which pays no norm tax at
all, holds to about 1% through half the induction period and then goes vacuous
at the branching knee. The conclusion is that the field's central query is
exponentially ill-conditioned, so a rigorous forward certificate on ignition
delay is expensive for reasons that are physical. The promising entry is a
quantity that is not exponentially amplified: the crossover limit separating
ignition from no ignition, which is an eigenvalue sign condition on a Metzler
matrix and therefore a cone-preserving bracket.

## Recurring certificate archetypes

Six patterns cover nearly everything above, and the rewrite library should be
organized around them.

1. **Small-parameter expansions.** Gyrokinetics, EFT, post-Newtonian,
   Braginskii. Asymptotic tier.
2. **Variational sandwiches.** Chemistry's upper and lower bracket, SOS
   transport bounds, reduced-basis a posteriori estimators. Rigorous tier.
3. **Resolution-limited queries.** HLT smearing, g−2 windows, GW mismatch. The
   ε is supplied by the instrument.
4. **Projection with memory.** Mori-Zwanzig in MD, plasma closures, subgrid
   models. Empirical tier, with variational scores where they exist.
5. **Cone-preserving brackets.** Reactor criticality, S_N transport. Rigorous
   tier.
6. **Existence certificates.** The pn junction's Newton-Kantorovich enclosure.
   Rigorous tier.

A compression contributed under one archetype in one field should transfer to
its siblings mechanically. That transfer is the flywheel working.

### The sixth archetype

The sixth is the only one that certifies something other than a number. Its
output is the statement that a solution of a nonlinear system exists, is
locally unique, and lies within a stated radius of the computed iterate. No
residual, however small, supports that claim on its own. Its siblings are every
nonlinear steady state a simulation reports as if it were a fact: chemical
equilibrium, structural post-buckling, periodic orbits, the operating points of
a circuit.

It shipped with one restriction. ‖J⁻¹‖ came from the M-matrix witness, so the
Jacobian had to be a Z-matrix. That restriction is now lifted, by the general
route of bounding ‖I − RJ‖ for an approximate inverse R, so that the Neumann
series for (RJ)⁻¹ converges and ‖J⁻¹‖ ≤ ‖R‖/(1 − ‖I − RJ‖). Nothing was traded
away for it. Measured on twelve junction Jacobians the two routes agree to
every digit printed, which is unsurprising once you notice that on an M-matrix
the witness is *exact* rather than merely valid, because J⁻¹ ≥ 0 makes
‖J⁻¹‖∞ literally ‖J⁻¹e‖∞. The cone route is still tried first, being a solve
rather than an inverse.

The lift immediately let the archetype transfer into a launch domain and remove
a refusal there. A tokamak's pressure peaks on the magnetic axis and falls away
outward, so a real profile is *decreasing* in ψ. That does two independent
things to the Grad-Shafranov problem.

The contraction factor the existing Prager-Synge certificate needs is
max|S′|·R_max/(R_min·λ₁). For an exponential profile of decay scale ψ₀ that
maximum is c/ψ₀ rather than c, so the wall arrives ψ₀ times sooner. Measured,
it refuses above c = 0.47, where a linear source reached 2.34.

Separately, differentiating the source contributes −S′ times the *consistent
mass matrix*, whose off-diagonals are integrals of products of non-negative
basis functions and are therefore positive. A decreasing profile puts them into
the Jacobian and closes the cone route at every c > 0, including couplings the
contraction would have accepted. So the Z-matrix lift was not optional here,
and it was not the coupling strength that demanded it.

Kantorovich by the Neumann route certifies out to c = 16, a contraction factor
of 32. What changes with the method is the guarantee. Prager-Synge measures the
distance to the continuum solution, and this measures the distance to the
discrete equilibrium. The mesh gap is printed rather than folded in. It is
second order, and twelve orders above the radius a converged rung certifies.

### The continuum step, shared by three domains

That gap is now closed, and closing it in three domains at once turned out to
be one function rather than three. The reactor, the junction and the tokamak
all cross the same line, from a certificate about a discretised model to one
about the thing modelled. They now cross it through `continuum_limit`. That
function reads the ladder with `gci_extrapolate`, adds the finest rung's own
error, and takes the weaker tier. It refuses when the rungs' brackets are
within a factor of ten of the gaps between them, because a convergence order
fitted to bracket noise is fiction.

The rule that falls out is now stated once instead of three times: **a
continuum answer is never RIGOROUS, however rigorous the rungs it is built
from.** It is also usually the answer that was wanted, which is the
uncomfortable half. The tokamak's discrete radius is 10⁻¹⁴ and its mesh gap is
10⁻⁴, so the tier labels rank the two claims in the opposite order to their
usefulness.

One thing was measured and deliberately not built. Splitting a single tolerance
between the mesh and the Newton iteration looks like the obvious fifth composed
plan, and it derives nothing, because the Newton half is free. At N=200 the
junction's certificate goes from 0.78 nC/cm², which is thirty-three times too
coarse to read the mesh trend, to 1.1·10⁻⁵ in one step. That is two thousand
times tighter than the ladder needs. Every budget in the range that matters
picks the same rung, so there is no exchange rate to price, and a stage split
would be decoration on a knob that does not respond. Quadratic convergence is
what does it.

### The fifth archetype

The fifth arrived last and is the one with the most unclaimed siblings. Where
an operator preserves a cone rather than a quadratic form, Perron-Frobenius
supplies the dominant eigenvalue and Collatz-Wielandt sandwiches it from any
strictly positive trial vector, with no symmetry required anywhere. Examples
are a population that cannot go negative, or a probability that cannot go
negative. The variational archetype cannot reach any of these, because none of
the operators is self-adjoint.

The transfer claim has been tested once rather than merely asserted. The same
witness and the same bracket, with no new proof and no new code, run on an S_N
discrete-ordinates transport operator as well as on the diffusion operator they
were written for. Step differencing keeps it inside the cone, which is the only
hypothesis either function cares about. Still unclaimed, and expected to go the
same way: Markov-state relaxation rates and transfer-operator spectra in
molecular kinetics.

## Engines, and the guess and check line

Criterion 4 says the compiler targets existing engines. The SOS work fixed the
shape of that relationship. cvxpy and SCS search, exact rational arithmetic
decides, and deleting the solver kills the search without weakening any
certificate. Every integration below sits on the same line. Engines propose.
Certificates are issued on our side of the line, and refusal stays available
whenever an engine's output cannot be checked at the declared tolerance.

**FEniCSx, the first bridge.** For elliptic problems, equilibrated-flux a
posteriori estimates (the Ern-Vohralík lineage) give bounds on the
discretization error that are guaranteed rather than estimated, and computable
from the discrete solution alone. Goal-oriented variants bound functionals.
That is a RIGOROUS-tier certificate for PDE functionals, which no resolution
ladder over a forward solver can provide at any budget. First artifact: a
certified Poisson functional with a guaranteed bound, verified on our side from
the fluxes FEniCSx returns. Then low-frequency Helmholtz, which connects back
to the wave-scattering column.

**Basilisk, the second bridge.** This is the nearest culture to this project:
adaptive refinement against a declared per-field tolerance, and a test suite of
literate pages that measure convergence orders in the open. It stops one step
short of certificates, because orders are reported and never enforced, and
there is no refusal. The bridge is to drive Basilisk as an untrusted external
binary, ladder its maximum refinement level, and mint grid-convergence
certificates on functionals. The level must be fixed per rung, because
Richardson over freely adapting meshes is not a ladder. First artifact: the
sea-wall triage rerun with the Saint-Venant solver in place of the toy WCSPH.
Second-order ladders should tighten the impulse certificate by an order of
magnitude, and may flip the low-berm refusal to a certificate. The cost is a C
toolchain (qcc), so recorded-run pages rather than CI regeneration, at least at
first.

**Reactor codes, and why not the obvious one.** The criticality checker takes
the loss and fission operators as plain arrays and asks four questions of them,
so the integration surface is already open. Any deterministic code willing to
export its assembled operators gets a certificate, with the discretization
staying that code's claim and the bracket staying ours. The obvious name,
OpenMC, is the wrong first bridge for exactly this reason. A continuous-energy
Monte Carlo code has no operator to hand over, and its k_eff arrives as a batch
standard deviation, which is a statistical statement of a different kind.
Bracketing that would mean wrapping its statistics at the EMPIRICAL tier rather
than applying the theorem. It should be built as that, honestly labelled,
rather than dressed up as the rigorous bracket beside it. A deterministic
transport code is the nearer bridge.

**Proposers per domain.** PySCF supplies integrals and reference wavefunctions
at scale while the Cholesky floors and window machinery stay ours. gwsurrogate
and bilby generate and consume waveforms around the conformal mismatch
certificate. QuTiP, stim and quimb are the engines the quantum dispatch
procedure chooses between. OpenMM, deeptime and pymbar carry molecular
kinetics, where MBAR's asymptotic variances pair naturally with the conformal
layer. Dedalus hosts the SOS transport program at PDE scale. pyrokinetics and
DESC sit under the plasma hierarchy dispatch.

**The checker side.** A short list, trusted only after declaration. python-flint
(Arb) ball arithmetic would harden the interval type and make degree-six
rational SOS cheap. mpmath already referees the test suite. The long game for
the ASYMPTOTIC tier's "exponent proven" clause is a proof assistant, and it can
wait.

Order of work: FEniCSx, then Basilisk, then PySCF. Rigor first, then the most
visible tightening, then scale.

## Sequencing

| Target | Roadmap phases | Dominant contraction | Certificate outlook |
|---|---|---|---|
| Wave scattering & CEM | 1 | rank structure, butterflies, reduced bases | rigorous / asymptotic |
| Plasmas & fusion | 2-4 | small-parameter hierarchy, closures | asymptotic to empirical |
| Quantum chemistry & materials | 2-3 | nearsightedness, low-rank ERIs, variational sandwich | rigorous intervals + declared-empirical |
| Lattice field theory | 2-3 | hierarchy, factorization, smearing | statistical + rigorous parts |
| Nuclear & dense matter | 2-3 | EFT expansion, reduced bases | statistical / asymptotic |
| Molecular kinetics & free energy | 4 | spectral-gap projection, Mori-Zwanzig | variational + declared-empirical |
| Quantum dynamics & devices | 2-4 | light cones, noise truncation, stabilizer rank | rigorous islands to empirical |
| Turbulence & transport | 4 | SOS bounds, subgrid closures | rigorous bounds + declared-empirical |
| Gravitational waves | 3 | ROM manifolds, PN/EOB expansions | empirical + asymptotic |

Beachhead order. CEM first, as the Phase 1 shakedown against MLFMA and
butterfly baselines. Then the chemistry energy bracket, as the first headline
rigorous interval. Then lattice windows and the plasma reduction hierarchy as
Phase 2 and 3 partners, chosen because both communities already budget error.
Then the quantum dispatch procedure and SOS transport bounds as certificate
showpieces. Molecular and plasma closure discovery come last, carrying Phase
4's declared-empirical banner. The selection logic throughout is to go where ε
is already spoken aloud, so the compiler's certificates land in a culture that
knows how to read them.

*Every phase of the roadmap should have a paying customer.*
